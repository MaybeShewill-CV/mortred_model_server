#!/usr/bin/env python3
"""Record SIGTERM stop order for one yolov8s + libface supervisor.

--check reads the unit file and a fixed sample trace. It does not start
processes, call strace, or create the live json.
The live run starts supervisor under strace in a new session, sends SIGTERM
only to that supervisor pid, and records kill/wait order from the trace.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVICE_PATH = ROOT / "deploy" / "mortred-supervisor.service"
RECORD_PATH = ROOT / "logs" / "stop_order_yolov8s_libface.json"
STRACE_BIN = "/usr/bin/strace"
SETSID_BIN = "/usr/bin/setsid"

YOLO_ID = "yolov8s"
FACE_ID = "libface"
GATEWAY_ID = "__gateway"
PID_KEYS = ("yolov8s", "libface", "gateway")

STATUS_PORT = 8787
PORTS = (8787, 8080, 9051, 9056)
# Existing autostart budgets. Do not extend them.
READY_BUDGET_S = 15 + 60
STOP_WAIT_S = 30
READY_POLL_S = 0.2

TOKEN_KEYS = (
    "MORTRED_API_TOKEN",
    "MORTRED_GATEWAY_AUTH_TOKEN",
    "MORTRED_METRICS_TOKEN",
)

_SECTION = (
    "worker_nums = 1\n"
    "gpu_mem_mib = 1800\n"
    "gpu_mem_at_workers = 1\n"
    'gpu_mem_source = "seed"\n'
)
PACK_TEXT = "[pack.yolov8s]\n" + _SECTION + "\n" + "[pack.libface]\n" + _SECTION

_SEED_LINES = (
    "worker_nums = 1",
    "gpu_mem_mib = 1800",
    "gpu_mem_at_workers = 1",
    'gpu_mem_source = "seed"',
)

CHECK_LINES = (
    "KillMode=process 为通过",
    "KillSignal=SIGTERM 为通过",
    "TimeoutStopSec=120 为通过",
    "模型 SIGINT 并等待之后才向 gateway 发 SIGINT 为通过",
    "gateway 的 SIGINT 早于任一模型退出为不通过",
    "未等到两个模型退出就向 gateway 发 SIGINT 为不通过",
    "gateway 退出后模型 pid 再次出现为不通过",
)
CHECK_STDOUT = "".join(line + "\n" for line in CHECK_LINES)

RECORD_KEYS = {
    "signal",
    "signal_pid",
    "supervisor_pid",
    "before",
    "pack",
    "pids",
    "ppid",
    "established_to_8080",
    "events",
    "after_gateway",
}
AFTER_KEYS = ("gateway_alive", "yolov8s_alive", "libface_alive")
KILL_CALLS = {"kill", "tgkill", "tkill", "rt_sigqueueinfo"}
WAIT_CALLS = {"wait4", "waitpid"}
LOOPBACK_8080 = "0100007F:1F90"

_PREFIX_RE = re.compile(
    r"^(?:\[pid\s+(\d+)\]\s+|(\d+)\s+)(\d+):(\d{2}):(\d{2})\.(\d+)\s+(.*)$"
)
_COMPLETE_RE = re.compile(r"^(\w+)\((.*)\)\s*=\s*(-?\d+)\b")
_UNFINISHED_RE = re.compile(r"^(\w+)\((.*)\s*<unfinished \.\.\.>\s*$")
_RESUMED_RE = re.compile(
    r"^<\.\.\.\s+(\w+)\s+resumed>\s*(.*)\)\s*=\s*(-?\d+)\b"
)

# strace -f prints the caller thread id. The fixed sample uses one thread id
# that is not the supervisor pid, matching the monitor thread that calls kill.
_SAMPLE_PIDS = {"yolov8s": 11, "libface": 22, "gateway": 33}
_SAMPLE_TIDS = {100, 101}
_GOOD_TRACE = "\n".join(
    (
        "999 00:00:01.000000 kill(11, SIGINT) = 0",
        "101 00:00:01.000000 kill(11, SIGINT) = 0",
        "101 00:00:01.000100 kill(22, 2) = 0",
        "101 00:00:01.000200 kill(11, SIGTERM) = 0",
        "101 00:00:01.010000 wait4(-1, NULL, WNOHANG, NULL) = 0",
        "101 00:00:01.100000 wait4(11,  <unfinished ...>",
        "101 00:00:01.150000 <... wait4 resumed>[{WIFSIGNALED(s) && WTERMSIG(s) == SIGINT}], WNOHANG, NULL) = 11",
        "101 00:00:01.160000 wait4(-1, [{WIFEXITED(s) && WEXITSTATUS(s) == 0}], WNOHANG, NULL) = 22",
        "101 00:00:01.200000 kill(33, SIGINT) = 0",
        "101 00:00:01.300000 waitpid(33, [{WIFEXITED(s) && WEXITSTATUS(s) == 0}], WNOHANG) = 33",
        "11 00:00:01.000050 --- SIGINT {si_signo=SIGINT, si_code=SI_USER, si_pid=100, si_uid=1000} ---",
        "101 00:00:01.310000 +++ exited with 0 +++",
    )
) + "\n"


def is_json_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def unit_ok(text: str) -> bool:
    """[Service] before [Install] has one exact KillMode=process line."""
    if not isinstance(text, str):
        return False
    if "ExecStop=" in text:
        return False
    lines = text.splitlines()
    if lines.count("[Service]") != 1 or lines.count("[Install]") != 1:
        return False
    service_at = lines.index("[Service]")
    install_at = lines.index("[Install]")
    if service_at >= install_at:
        return False
    span = lines[service_at + 1 : install_at]
    if span.count("KillMode=process") != 1:
        return False
    if [line for line in lines if "KillMode" in line] != ["KillMode=process"]:
        return False
    if [line for line in lines if "KillSignal" in line] != ["KillSignal=SIGTERM"]:
        return False
    if [line for line in lines if "TimeoutStopSec" in line] != ["TimeoutStopSec=120"]:
        return False
    return True


def _minimal_unit() -> str:
    return (
        "[Service]\n"
        "TimeoutStopSec=120\n"
        "KillSignal=SIGTERM\n"
        "KillMode=process\n"
        "[Install]\n"
    )


def unit_rules_hold(real_text: str) -> bool:
    if not unit_ok(real_text) or not unit_ok(_minimal_unit()):
        return False
    valid = _minimal_unit()
    rejected = (
        valid.replace("KillMode=process", "KillMode=mixed"),
        valid.replace("KillMode=process", "KillMode=control-group"),
        valid.replace("KillMode=process", "KillMode=none"),
        valid.replace("KillMode=process", "KillMode = process"),
        valid.replace("KillSignal=SIGTERM", "KillSignal = SIGTERM"),
        valid.replace("TimeoutStopSec=120", "TimeoutStopSec=90"),
        valid.replace("KillMode=process\n", "KillMode=process\nKillMode=process\n"),
        valid.replace("KillMode=process\n", ""),
        valid + "ExecStop=/bin/true\n",
        "[Install]\nKillMode=process\n[Service]\n"
        "TimeoutStopSec=120\nKillSignal=SIGTERM\n",
    )
    return all(not unit_ok(item) for item in rejected)


def one_server(status: dict, model_id: str):
    servers = status.get("servers")
    if not isinstance(servers, list):
        return None
    found = [
        item
        for item in servers
        if isinstance(item, dict) and item.get("id") == model_id
    ]
    if len(found) != 1:
        return None
    return found[0]


def trio(status):
    if not isinstance(status, dict):
        return None
    yolo = one_server(status, YOLO_ID)
    face = one_server(status, FACE_ID)
    gateway = status.get("gateway")
    if yolo is None or face is None:
        return None
    if not isinstance(gateway, dict) or gateway.get("id") != GATEWAY_ID:
        return None
    return yolo, face, gateway


def place_running(obj: dict) -> bool:
    if obj.get("ready") is not True:
        return False
    if obj.get("state") != "running":
        return False
    pid = obj.get("pid")
    return is_json_int(pid) and pid > 0


def pack_ok(text) -> bool:
    if not isinstance(text, str):
        return False
    if "model_config" in text or "decode_auto" in text:
        return False
    if text.count("[pack.") != 2:
        return False
    if "[pack.yolov8s]" not in text or "[pack.libface]" not in text:
        return False
    parts = text.split("[pack.")
    if len(parts) != 3:
        return False
    names = []
    for part in parts[1:]:
        end = part.find("]")
        if end < 0:
            return False
        names.append(part[:end])
        body_lines = [line.strip() for line in part[end + 1 :].splitlines()]
        if any(line not in body_lines for line in _SEED_LINES):
            return False
    return names == ["yolov8s", "libface"] or names == ["libface", "yolov8s"]


def _ts_us(hour: str, minute: str, second: str, frac: str) -> int:
    if len(frac) > 6:
        usec = int(frac[:6])
    else:
        usec = int(frac) * (10 ** (6 - len(frac)))
    return ((int(hour) * 60 + int(minute)) * 60 + int(second)) * 1000000 + usec


def _split_top(args: str):
    parts = []
    buf = []
    depth = 0
    for ch in args:
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _first_int(text: str):
    match = re.match(r"-?\d+", text.strip())
    if match is None:
        return None
    return int(match.group(0))


def _is_sigint(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    token = stripped.split()[0].rstrip(",")
    return token in {"SIGINT", "2", "0x2", "0X2"}


def _kill_target_signal(name: str, args: str):
    parts = _split_top(args)
    if name == "kill" and len(parts) >= 2:
        return [_first_int(parts[0])], parts[1]
    if name == "tkill" and len(parts) >= 2:
        return [_first_int(parts[0])], parts[1]
    if name == "tgkill" and len(parts) >= 3:
        return [_first_int(parts[0]), _first_int(parts[1])], parts[2]
    if name == "rt_sigqueueinfo" and len(parts) >= 2:
        return [_first_int(parts[0])], parts[1]
    return [], ""


def _event(pid_key: str, op: str, ts_us: int, sig: str | None = None) -> dict:
    item = {"pid_key": pid_key, "op": op, "ts_us": ts_us}
    if op == "kill":
        item = {
            "pid_key": pid_key,
            "op": op,
            "signal": sig,
            "ts_us": ts_us,
        }
    return item


def _emit(name, args, ret, caller, ts_us, supervisor_tids, key_by_pid, out):
    if caller not in supervisor_tids and name in KILL_CALLS:
        return
    if name in KILL_CALLS:
        if ret != 0:
            return
        targets, sig = _kill_target_signal(name, args)
        if not _is_sigint(sig):
            return
        seen = set()
        for target in targets:
            key = key_by_pid.get(target)
            if key is None or key in seen:
                continue
            seen.add(key)
            out.append(_event(key, "kill", ts_us, "SIGINT"))
        return
    if name in WAIT_CALLS and ret in key_by_pid:
        out.append(_event(key_by_pid[ret], "wait", ts_us))


def parse_trace(text: str, supervisor_tids, pids: dict) -> list | None:
    """Keep supervisor kill calls and waits whose return pid is one of the three."""
    if not isinstance(text, str):
        return None
    if not isinstance(supervisor_tids, set) or not supervisor_tids:
        return None
    key_by_pid = {}
    for key in PID_KEYS:
        pid = pids.get(key)
        if not is_json_int(pid) or pid <= 0 or pid in key_by_pid:
            return None
        key_by_pid[pid] = key
    events = []
    pending = {}
    for raw in text.splitlines():
        match = _PREFIX_RE.match(raw.strip())
        if match is None:
            continue
        caller_text = match.group(1) or match.group(2)
        caller = int(caller_text)
        ts_us = _ts_us(match.group(3), match.group(4), match.group(5), match.group(6))
        body = match.group(7)
        if body.startswith("---") or body.startswith("+++"):
            continue
        resumed = _RESUMED_RE.match(body)
        if resumed is not None:
            name = resumed.group(1)
            args = pending.pop((caller, name), resumed.group(2))
            ret = int(resumed.group(3))
            _emit(name, args, ret, caller, ts_us, supervisor_tids, key_by_pid, events)
            continue
        unfinished = _UNFINISHED_RE.match(body)
        if unfinished is not None:
            pending[(caller, unfinished.group(1))] = unfinished.group(2).strip()
            continue
        complete = _COMPLETE_RE.match(body)
        if complete is None:
            continue
        _emit(
            complete.group(1),
            complete.group(2),
            int(complete.group(3)),
            caller,
            ts_us,
            supervisor_tids,
            key_by_pid,
            events,
        )
    return events


def choose_events(events):
    """One SIGINT kill and a later wait per model, then a later gateway SIGINT."""
    if not isinstance(events, list):
        return None
    grouped = {key: {"kill": [], "wait": []} for key in PID_KEYS}
    for item in events:
        if not isinstance(item, dict):
            return None
        key = item.get("pid_key")
        op = item.get("op")
        ts_us = item.get("ts_us")
        if key not in grouped or op not in {"kill", "wait"}:
            return None
        if not is_json_int(ts_us):
            return None
        if op == "kill":
            if item.get("signal") != "SIGINT":
                return None
            if set(item) != {"pid_key", "op", "signal", "ts_us"}:
                return None
        elif set(item) != {"pid_key", "op", "ts_us"}:
            return None
        grouped[key][op].append(item)
    chosen = []
    model_waits = {}
    for key in (YOLO_ID, FACE_ID):
        kills = grouped[key]["kill"]
        if not kills:
            return None
        kill = min(kills, key=lambda item: item["ts_us"])
        later = [item for item in grouped[key]["wait"] if item["ts_us"] > kill["ts_us"]]
        if not later:
            return None
        wait = min(later, key=lambda item: item["ts_us"])
        chosen.append(kill)
        chosen.append(wait)
        model_waits[key] = wait
    gate_kills = [
        item
        for item in grouped["gateway"]["kill"]
        if item["ts_us"] > model_waits[YOLO_ID]["ts_us"]
        and item["ts_us"] > model_waits[FACE_ID]["ts_us"]
    ]
    if not gate_kills:
        return None
    gate_kill = min(gate_kills, key=lambda item: item["ts_us"])
    chosen.append(gate_kill)
    later_gate = [
        item for item in grouped["gateway"]["wait"] if item["ts_us"] > gate_kill["ts_us"]
    ]
    if later_gate:
        chosen.append(min(later_gate, key=lambda item: item["ts_us"]))
    chosen.sort(key=lambda item: (item["ts_us"], 0 if item["op"] == "kill" else 1, item["pid_key"]))
    return chosen


def count_established(text: str) -> int | None:
    """Count established IPv4 sockets whose local address is 127.0.0.1:8080."""
    if not isinstance(text, str):
        return None
    count = 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        if parts[1].upper() == LOOPBACK_8080 and parts[3].upper() == "01":
            count += 1
    return count


def _pid_map_ok(value, supervisor_pid: int) -> bool:
    if not isinstance(value, dict) or set(value) != set(PID_KEYS):
        return False
    for key in PID_KEYS:
        if not is_json_int(value.get(key)) or value[key] != supervisor_pid:
            return False
    return True


def _after_ok(value) -> bool:
    if not isinstance(value, dict) or set(value) != set(AFTER_KEYS):
        return False
    for key in AFTER_KEYS:
        if value.get(key) is not False:
            return False
    return True


def judge_record(record) -> bool:
    """Acceptance shape for the json written by a passing live run."""
    if not isinstance(record, dict) or set(record) != RECORD_KEYS:
        return False
    if record.get("signal") != "SIGTERM":
        return False
    signal_pid = record.get("signal_pid")
    supervisor_pid = record.get("supervisor_pid")
    if not is_json_int(signal_pid) or not is_json_int(supervisor_pid):
        return False
    if signal_pid <= 0 or signal_pid != supervisor_pid:
        return False
    established = record.get("established_to_8080")
    if not is_json_int(established) or established != 0:
        return False
    if choose_events(record.get("events")) is None:
        return False
    if not _after_ok(record.get("after_gateway")):
        return False
    if not pack_ok(record.get("pack")):
        return False
    found = trio(record.get("before"))
    if found is None or not all(place_running(obj) for obj in found):
        return False
    yolo, face, gateway = found
    pids = record.get("pids")
    if not isinstance(pids, dict) or set(pids) != set(PID_KEYS):
        return False
    if pids.get("yolov8s") != yolo.get("pid"):
        return False
    if pids.get("libface") != face.get("pid"):
        return False
    if pids.get("gateway") != gateway.get("pid"):
        return False
    if not _pid_map_ok(record.get("ppid"), supervisor_pid):
        return False
    text = json.dumps(record, ensure_ascii=False)
    if "systemctl" in text:
        return False
    return True


def _trace_with_gateway_at(clock: str) -> str:
    return _GOOD_TRACE.replace("00:00:01.200000 kill(33, SIGINT)", "%s kill(33, SIGINT)" % clock)


def _synthetic_record(events):
    before = {
        "gateway": {"id": GATEWAY_ID, "state": "running", "pid": 33, "ready": True},
        "servers": [
            {"id": YOLO_ID, "state": "running", "pid": 11, "ready": True},
            {"id": FACE_ID, "state": "running", "pid": 22, "ready": True},
            {"id": "mobilenetv2", "state": "stopped", "pid": 0, "ready": False},
        ],
    }
    return {
        "signal": "SIGTERM",
        "signal_pid": 100,
        "supervisor_pid": 100,
        "before": before,
        "pack": PACK_TEXT,
        "pids": {"yolov8s": 11, "libface": 22, "gateway": 33},
        "ppid": {"yolov8s": 100, "libface": 100, "gateway": 100},
        "established_to_8080": 0,
        "events": events,
        "after_gateway": {
            "gateway_alive": False,
            "yolov8s_alive": False,
            "libface_alive": False,
        },
    }


def _tcp_rules_hold() -> bool:
    listen_only = (
        "  sl  local_address rem_address   st\n"
        "   0: 0100007F:1F90 00000000:0000 0A 00000000:00000000\n"
        "   1: 0100007F:226F 0100007F:1F90 01 00000000:00000000\n"
    )
    one_established = (
        listen_only
        + "   2: 0100007F:1F90 0100007F:C000 01 00000000:00000000\n"
    )
    if count_established(listen_only) != 0:
        return False
    if count_established(one_established) != 1:
        return False
    if count_established("") != 0:
        return False
    return True


def order_rules_hold() -> bool:
    if not pack_ok(PACK_TEXT):
        return False
    if pack_ok(PACK_TEXT + "[pack.mobilenetv2]\n"):
        return False
    if pack_ok(PACK_TEXT.replace("worker_nums = 1", "decode_auto = true", 1)):
        return False
    if not _tcp_rules_hold():
        return False
    good = parse_trace(_GOOD_TRACE, set(_SAMPLE_TIDS), _SAMPLE_PIDS)
    chosen = choose_events(good)
    if chosen is None or not judge_record(_synthetic_record(chosen)):
        return False
    kill_callers = []
    for item in chosen:
        if item["op"] == "kill" and item["pid_key"] in {YOLO_ID, FACE_ID}:
            kill_callers.append(item["ts_us"])
    if len(kill_callers) != 2:
        return False
    # A trace whose only SIGINT kills come from outside the supervisor is not an order.
    outsider = parse_trace(_GOOD_TRACE, {999}, _SAMPLE_PIDS)
    if choose_events(outsider) is not None:
        return False
    early = parse_trace(_trace_with_gateway_at("00:00:01.000050"), set(_SAMPLE_TIDS), _SAMPLE_PIDS)
    if choose_events(early) is not None:
        return False
    one_model = parse_trace(
        _trace_with_gateway_at("00:00:01.155000"), set(_SAMPLE_TIDS), _SAMPLE_PIDS
    )
    if choose_events(one_model) is not None:
        return False
    same_time = parse_trace(
        _trace_with_gateway_at("00:00:01.160000"), set(_SAMPLE_TIDS), _SAMPLE_PIDS
    )
    if choose_events(same_time) is not None:
        return False
    tgkill_trace = (
        "101 00:00:01.000000 tgkill(11, 11, SIGINT) = 0\n"
        "101 00:00:01.000100 tkill(22, SIGINT) = 0\n"
        "101 00:00:01.150000 wait4(-1, [{WIFEXITED(s) && WEXITSTATUS(s) == 0}], 0, NULL) = 11\n"
        "101 00:00:01.160000 wait4(-1, [{WIFEXITED(s) && WEXITSTATUS(s) == 0}], 0, NULL) = 22\n"
        "101 00:00:01.200000 rt_sigqueueinfo(33, SIGINT, {si_signo=SIGINT}) = 0\n"
        "101 00:00:01.300000 wait4(-1, [{WIFEXITED(s) && WEXITSTATUS(s) == 0}], 0, NULL) = 33\n"
    )
    if choose_events(parse_trace(tgkill_trace, set(_SAMPLE_TIDS), _SAMPLE_PIDS)) is None:
        return False
    record = _synthetic_record(chosen)
    alive = _synthetic_record(chosen)
    alive["after_gateway"] = dict(record["after_gateway"])
    alive["after_gateway"]["yolov8s_alive"] = True
    connected = _synthetic_record(chosen)
    connected["established_to_8080"] = 1
    bad_signal = _synthetic_record(chosen)
    bad_signal["signal"] = "SIGINT"
    bad_pid = _synthetic_record(chosen)
    bad_pid["signal_pid"] = bad_pid["supervisor_pid"] + 1
    bad_ppid = _synthetic_record(chosen)
    bad_ppid["ppid"] = {"yolov8s": 100, "libface": 100, "gateway": 101}
    bad_ready = _synthetic_record(chosen)
    bad_ready["before"] = {
        "gateway": dict(record["before"]["gateway"]),
        "servers": [dict(item) for item in record["before"]["servers"]],
    }
    bad_ready["before"]["servers"][0]["ready"] = False
    bad_word = _synthetic_record(chosen)
    bad_word["before"] = dict(record["before"])
    bad_word["before"]["note"] = "systemctl"
    with_pid = _synthetic_record(chosen)
    with_pid["after_gateway"] = dict(record["after_gateway"])
    with_pid["after_gateway"]["pid"] = 11
    if any(
        judge_record(item)
        for item in (
            alive,
            connected,
            bad_signal,
            bad_pid,
            bad_ppid,
            bad_ready,
            bad_word,
            with_pid,
        )
    ):
        return False
    return True


def run_check() -> int:
    try:
        text = SERVICE_PATH.read_text(encoding="utf-8")
    except OSError:
        return _fail("unit file unreadable\n")
    if not unit_rules_hold(text) or not order_rules_hold():
        return _fail("stop order check did not match\n")
    sys.stdout.buffer.write(CHECK_STDOUT.encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


def _fail(message: str) -> int:
    if not message.endswith("\n"):
        message += "\n"
    sys.stderr.buffer.write(message.encode("utf-8"))
    return 1


def port_in_use(port: int) -> bool:
    # bind only. This does not connect, including not to 127.0.0.1:8080.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        return True
    finally:
        sock.close()
    return False


def load_tokens():
    found = {}
    for key in TOKEN_KEYS:
        value = os.environ.get(key, "")
        if value:
            found[key] = value
    missing = [key for key in TOKEN_KEYS if key not in found]
    if missing:
        path = ROOT / "conf" / "local" / "trust.env"
        if path.is_file():
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'").strip('"')
                if key in TOKEN_KEYS and key not in found and value:
                    found[key] = value
    values = [found.get(key, "") for key in TOKEN_KEYS]
    if any(not value for value in values):
        return None
    if len(set(values)) != 3:
        return None
    return {key: found[key] for key in TOKEN_KEYS}


def find_supervisor():
    for rel in ("_bin/mortred-supervisor.out", "bin/mortred-supervisor.out"):
        path = ROOT / rel
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


def exchange(host, port, method, path, token, body, timeout):
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    if body is not None:
        headers["Content-Type"] = "application/json"
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        payload = resp.read()
        return resp.status, payload
    finally:
        conn.close()


def fetch_status(token):
    try:
        status, payload = exchange(
            "127.0.0.1", STATUS_PORT, "GET", "/api/v1/status", token, None, 5
        )
    except (OSError, http.client.HTTPException):
        return None
    if status != 200:
        return None
    try:
        doc = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict):
        return None
    return doc


def trio_ready(doc):
    found = trio(doc)
    if found is None or not all(place_running(obj) for obj in found):
        return None
    return found


def child_env(pack_path: Path, tokens: dict) -> dict:
    env = os.environ.copy()
    env["MORTRED_PACK"] = str(pack_path)
    # Symlinked _bin resolves /proc/self/exe outside this worktree.
    env["MORTRED_PROJECT_ROOT"] = str(ROOT)
    for key, value in tokens.items():
        env[key] = value
    prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = "%s:%s:%s" % (
        ROOT / "_lib",
        ROOT / "3rd_party" / "libs",
        prev,
    )
    return env


def pid_alive(pid: int) -> bool:
    if not is_json_int(pid) or pid <= 0:
        return False
    return (Path("/proc") / str(pid)).exists()


def read_ppid(pid: int):
    path = Path("/proc") / str(pid) / "status"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if not line.startswith("PPid:"):
            continue
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            value = int(parts[1])
            if value > 0:
                return value
        return None
    return None


def read_sid(pid: int):
    path = Path("/proc") / str(pid) / "stat"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    end = text.rfind(")")
    if end < 0:
        return None
    fields = text[end + 2 :].split()
    if len(fields) < 4 or not fields[3].isdigit():
        return None
    return int(fields[3])


def read_tids(pid: int):
    task_dir = Path("/proc") / str(pid) / "task"
    tids = set()
    try:
        for entry in task_dir.iterdir():
            if entry.name.isdigit():
                tids.add(int(entry.name))
    except OSError:
        return None
    tids.add(pid)
    return tids


def read_exe(pid: int) -> str:
    try:
        return os.readlink("/proc/%d/exe" % pid)
    except OSError:
        return ""


def child_pids(pid: int):
    path = Path("/proc") / str(pid) / "task" / str(pid) / "children"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    found = []
    for part in text.split():
        if part.isdigit():
            found.append(int(part))
    return found


def find_traced_supervisor(strace_pid: int, binary: Path):
    expected = os.path.realpath(str(binary))
    base = os.path.basename(expected)
    for child in child_pids(strace_pid):
        exe = read_exe(child)
        if exe == expected or exe.endswith("/" + base):
            return child
    return None


def read_established_file() -> int | None:
    path = Path("/proc/net/tcp")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return count_established(text)


def wait_for_exit(pid: int, timeout_s: float) -> bool:
    if not pid_alive(pid):
        return True
    try:
        fd = os.pidfd_open(pid)
    except (AttributeError, OSError):
        return not pid_alive(pid)
    try:
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        hit = poller.poll(int(timeout_s * 1000))
        return bool(hit) or not pid_alive(pid)
    finally:
        os.close(fd)


def signal_supervisor(pid: int) -> bool:
    # This pid only. Not the process group and not a child pid.
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    return True


def redact(text: str, tokens: dict) -> str:
    for value in tokens.values():
        if value:
            text = text.replace(value, "***")
    return text


def note_log(state, tokens) -> None:
    tmpdir = state.get("tmpdir")
    if not tmpdir:
        return
    log_fp = state.get("log_fp")
    if log_fp is not None:
        try:
            log_fp.flush()
        except OSError:
            pass
    path = Path(tmpdir) / "supervisor.log"
    try:
        data = path.read_bytes()
    except OSError:
        return
    text = redact(data.decode("utf-8", errors="replace"), tokens)
    tail = "\n".join(text.splitlines()[-40:])
    if tail:
        sys.stderr.buffer.write((tail + "\n").encode("utf-8"))


def write_record(record: dict, tokens: dict) -> bool:
    text = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    if "systemctl" in text:
        return False
    for value in tokens.values():
        if value and value in text:
            return False
    try:
        RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = RECORD_PATH.with_name(RECORD_PATH.name + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp, RECORD_PATH)
    except OSError:
        return False
    try:
        written = RECORD_PATH.read_text(encoding="utf-8")
    except OSError:
        return False
    if written != text:
        return False
    if "systemctl" in written:
        return False
    for value in tokens.values():
        if value and value in written:
            try:
                RECORD_PATH.unlink()
            except OSError:
                pass
            return False
    return True


def _describe_events(events) -> str:
    if not isinstance(events, list):
        return "no events"
    lines = []
    for item in events:
        if not isinstance(item, dict):
            continue
        if item.get("op") == "kill":
            lines.append(
                "%s kill %s %s"
                % (item.get("pid_key"), item.get("signal"), item.get("ts_us"))
            )
        else:
            lines.append("%s wait %s" % (item.get("pid_key"), item.get("ts_us")))
    return "\n".join(lines)


def _relevant_trace(text: str) -> str:
    kept = []
    for line in text.splitlines():
        if not any(name in line for name in ("kill(", "tgkill(", "tkill(", "rt_sigqueueinfo(", "wait4(", "waitpid(")):
            continue
        if "wait" in line and (line.rstrip().endswith("= 0") or " = -1 " in line):
            continue
        kept.append(line)
        if len(kept) >= 80:
            break
    return "\n".join(kept)


def _run_live(state: dict) -> int:
    try:
        unit_text = SERVICE_PATH.read_text(encoding="utf-8")
    except OSError:
        return _fail("unit file unreadable\n")
    if not unit_ok(unit_text):
        return _fail("unit file does not match the stop contract\n")
    if not pack_ok(PACK_TEXT):
        return _fail("temporary pack text does not match\n")
    if not os.path.isfile(STRACE_BIN) or not os.access(STRACE_BIN, os.X_OK):
        return _fail("strace not found\n")
    if not os.path.isfile(SETSID_BIN) or not os.access(SETSID_BIN, os.X_OK):
        return _fail("setsid not found\n")
    binary = find_supervisor()
    if binary is None:
        return _fail("mortred-supervisor.out not found\n")
    tokens = load_tokens()
    if tokens is None:
        return _fail("management, inference, or metrics token missing or not distinct\n")
    state["tokens"] = tokens
    busy = [port for port in PORTS if port_in_use(port)]
    if busy:
        listed = " ".join(str(port) for port in busy)
        return _fail("port already in use: %s\n" % listed)

    tmpdir = tempfile.mkdtemp(prefix="mortred-stop-order-")
    state["tmpdir"] = tmpdir
    pack_path = Path(tmpdir) / "pack.toml"
    pack_path.write_bytes(PACK_TEXT.encode("utf-8"))
    state["pack_path"] = pack_path
    pack_text = pack_path.read_text(encoding="utf-8")
    if pack_text != PACK_TEXT or not pack_ok(pack_text):
        return _fail("temporary pack text does not match\n")
    trace_path = str(Path(tmpdir) / "strace.txt")
    state["trace_path"] = trace_path

    log_fp = open(Path(tmpdir) / "supervisor.log", "wb")
    state["log_fp"] = log_fp
    proc = subprocess.Popen(
        [
            STRACE_BIN,
            "-f",
            "-tt",
            "-e",
            "trace=kill,tgkill,tkill,rt_sigqueueinfo,wait4,waitpid",
            "-o",
            trace_path,
            SETSID_BIN,
            str(binary),
        ],
        cwd=str(ROOT),
        env=child_env(pack_path, tokens),
        stdin=subprocess.DEVNULL,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    state["proc"] = proc

    deadline = time.monotonic() + READY_BUDGET_S
    before = None
    supervisor_pid = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return _fail("supervisor exited before both models were ready\n")
        if supervisor_pid is None:
            supervisor_pid = find_traced_supervisor(proc.pid, binary)
            if supervisor_pid is not None:
                state["supervisor_pid"] = supervisor_pid
        if supervisor_pid is not None:
            doc = fetch_status(tokens["MORTRED_API_TOKEN"])
            if doc is not None and trio_ready(doc) is not None:
                before = doc
                break
        time.sleep(READY_POLL_S)
    if before is None or supervisor_pid is None:
        return _fail("yolov8s and libface were not both ready\n")

    if read_sid(supervisor_pid) != supervisor_pid:
        return _fail("supervisor is not in a new session\n")
    found = trio_ready(before)
    yolo, face, gateway = found
    yolo_pid = yolo["pid"]
    face_pid = face["pid"]
    gateway_pid = gateway["pid"]
    if not is_json_int(supervisor_pid) or supervisor_pid <= 0:
        return _fail("supervisor pid is not a positive integer\n")
    pids = {
        "yolov8s": yolo_pid,
        "libface": face_pid,
        "gateway": gateway_pid,
    }
    ppid = {}
    for name, pid in pids.items():
        parent = read_ppid(pid)
        if parent != supervisor_pid:
            return _fail("child ppid does not match supervisor\n")
        ppid[name] = parent
    tids = read_tids(supervisor_pid)
    if not tids or supervisor_pid not in tids:
        return _fail("supervisor threads were not read\n")

    established = read_established_file()
    if established is None:
        return _fail("could not read established connections\n")
    if not signal_supervisor(supervisor_pid):
        return _fail("could not signal supervisor\n")
    state["signaled"] = True
    if not wait_for_exit(supervisor_pid, STOP_WAIT_S):
        return _fail("supervisor did not exit\n")
    remaining = STOP_WAIT_S
    try:
        proc.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        return _fail("strace did not exit\n")

    try:
        trace_text = Path(trace_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return _fail("strace output unreadable\n")
    events = parse_trace(trace_text, tids, pids)
    chosen = choose_events(events)
    if chosen is None:
        sys.stderr.buffer.write(
            ("stop order events\n%s\n%s\n" % (_describe_events(events), _relevant_trace(trace_text))).encode(
                "utf-8"
            )
        )
        return _fail("stop order did not pass\n")
    gate_kill = [
        item for item in chosen if item["pid_key"] == "gateway" and item["op"] == "kill"
    ]
    gate_wait = [
        item
        for item in chosen
        if item["pid_key"] == "gateway" and item["op"] == "wait" and item["ts_us"] > gate_kill[0]["ts_us"]
    ]
    if not gate_wait:
        return _fail("gateway wait was not in the trace\n")
    after_gateway = {
        "gateway_alive": pid_alive(gateway_pid),
        "yolov8s_alive": pid_alive(yolo_pid),
        "libface_alive": pid_alive(face_pid),
    }
    record = {
        "signal": "SIGTERM",
        "signal_pid": supervisor_pid,
        "supervisor_pid": supervisor_pid,
        "before": before,
        "pack": pack_text,
        "pids": pids,
        "ppid": ppid,
        "established_to_8080": established,
        "events": chosen,
        "after_gateway": after_gateway,
    }
    if not judge_record(record):
        sys.stderr.buffer.write(
            ("stop order events\n%s\n" % _describe_events(chosen)).encode("utf-8")
        )
        return _fail("stop order record did not pass\n")
    if not write_record(record, tokens):
        return _fail("refusing to write stop order record\n")
    return 0


def stop_process(state: dict) -> bool:
    proc = state.get("proc")
    supervisor_pid = state.get("supervisor_pid")
    if supervisor_pid is None and proc is not None and proc.poll() is None:
        kids = child_pids(proc.pid)
        if kids:
            supervisor_pid = kids[0]
            state["supervisor_pid"] = supervisor_pid
    if supervisor_pid and pid_alive(supervisor_pid):
        if not state.get("signaled"):
            if not signal_supervisor(supervisor_pid):
                return False
            state["signaled"] = True
        if not wait_for_exit(supervisor_pid, STOP_WAIT_S):
            try:
                os.kill(supervisor_pid, signal.SIGKILL)
            except OSError:
                return False
            if not wait_for_exit(supervisor_pid, 5):
                return False
            return False
    if proc is None:
        return True
    if proc.poll() is not None:
        return True
    try:
        proc.wait(timeout=STOP_WAIT_S)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return False
    return True


def remove_temp(state: dict) -> bool:
    log_fp = state.get("log_fp")
    if log_fp is not None:
        try:
            log_fp.close()
        except OSError:
            return False
    tmpdir = state.get("tmpdir")
    if not tmpdir:
        return True
    ok = True
    try:
        for child in Path(tmpdir).iterdir():
            try:
                child.unlink()
            except OSError:
                ok = False
        os.rmdir(tmpdir)
    except OSError:
        ok = False
    return ok


def run_live() -> int:
    state = {
        "proc": None,
        "pack_path": None,
        "tmpdir": None,
        "log_fp": None,
        "signaled": False,
        "tokens": {},
        "supervisor_pid": None,
        "trace_path": None,
    }
    code = 1
    try:
        code = _run_live(state)
    except Exception:
        sys.stderr.buffer.write(b"stop order run failed\n")
        code = 1
    finally:
        if code != 0:
            note_log(state, state.get("tokens") or {})
        process_stopped = stop_process(state)
        removed = remove_temp(state)
        if not process_stopped:
            sys.stderr.buffer.write(b"supervisor did not exit\n")
            code = 1
        if not removed:
            sys.stderr.buffer.write(b"temp pack cleanup failed\n")
            code = 1
    return code


def main(argv: list[str]) -> int:
    if argv == ["--check"]:
        return run_check()
    if argv == []:
        return run_live()
    sys.stderr.buffer.write(b"usage: python3 scripts/stop_order_record.py [--check]\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
