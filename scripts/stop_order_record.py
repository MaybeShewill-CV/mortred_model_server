#!/usr/bin/env python3
"""Record SIGTERM stop order for one yolov8s + libface supervisor.

--check and the live run share unit_ok() and the exit-order judgment.
--check does not start processes and does not create the live json.
The last four --check lines are fixed sentences.
The live run keeps one unread gateway TCP session until the models_gone
sample so the existing server stop can still be seen at 50 ms.
"""

from __future__ import annotations

import copy
import http.client
import json
import os
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

YOLO_ID = "yolov8s"
FACE_ID = "libface"
GATEWAY_ID = "__gateway"

STATUS_PORT = 8787
PORTS = (8787, 8080, 9051, 9056)
GATEWAY_PORT = 8080
# Existing autostart budgets. Do not extend them.
READY_BUDGET_S = 15 + 60
STOP_WAIT_S = 30
SAMPLE_INTERVAL_S = 0.05

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
    "模型先退出且当时 gateway 与 supervisor 仍在为通过",
    "gateway 先于任一模型消失为不通过",
    "模型消失时 supervisor 已退出为不通过",
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
    "models_gone",
    "gateway_gone",
    "after_supervisor",
}
PID_KEYS = ("yolov8s", "libface", "gateway")
SAMPLE_KEYS = (
    "sample_index",
    "supervisor_alive",
    "gateway_alive",
    "yolov8s_alive",
    "libface_alive",
)
ALIVE_KEYS = SAMPLE_KEYS[1:]


def is_json_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def unit_ok(text: str) -> bool:
    """[Service] before [Install] has one exact KillMode=process line."""
    if not isinstance(text, str):
        return False
    if "ExecStop=" in text or "SendSIGKILL=" in text or "FinalKillSignal=" in text:
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
        valid + "SendSIGKILL=no\n",
        valid + "FinalKillSignal=SIGKILL\n",
        "[Install]\nKillMode=process\n[Service]\n"
        "TimeoutStopSec=120\nKillSignal=SIGTERM\n",
    )
    return all(not unit_ok(item) for item in rejected)


def sample_well_formed(sample) -> bool:
    if not isinstance(sample, dict) or set(sample) != set(SAMPLE_KEYS):
        return False
    if not is_json_int(sample.get("sample_index")) or sample["sample_index"] < 0:
        return False
    for key in ALIVE_KEYS:
        if type(sample.get(key)) is not bool:
            return False
    return True


def milestones_ok(models_gone, gateway_gone, after) -> bool:
    if not all(sample_well_formed(item) for item in (models_gone, gateway_gone, after)):
        return False
    if models_gone["yolov8s_alive"] or models_gone["libface_alive"]:
        return False
    if not models_gone["gateway_alive"] or not models_gone["supervisor_alive"]:
        return False
    if (
        gateway_gone["gateway_alive"]
        or gateway_gone["yolov8s_alive"]
        or gateway_gone["libface_alive"]
    ):
        return False
    if gateway_gone["sample_index"] <= models_gone["sample_index"]:
        return False
    if (
        after["supervisor_alive"]
        or after["gateway_alive"]
        or after["yolov8s_alive"]
        or after["libface_alive"]
    ):
        return False
    if after["sample_index"] <= gateway_gone["sample_index"]:
        return False
    return True


def _parse_samples(samples):
    if not isinstance(samples, list) or not samples:
        return None
    parsed = []
    expected = None
    for sample in samples:
        if not sample_well_formed(sample):
            return None
        index = sample["sample_index"]
        if expected is None:
            expected = index
        elif index != expected:
            return None
        expected = index + 1
        parsed.append(sample)
    return parsed


def _copy_sample(sample: dict) -> dict:
    return {key: sample[key] for key in SAMPLE_KEYS}


def _finish(models_gone, gateway_gone, after):
    if not milestones_ok(models_gone, gateway_gone, after):
        return "fail", None
    return "pass", (
        _copy_sample(models_gone),
        _copy_sample(gateway_gone),
        _copy_sample(after),
    )


def classify_order(samples):
    """Walk 50 ms samples. Return pass plus three objects, or a failure tag.

    pending means the trace has not reached a decision yet.
    """
    parsed = _parse_samples(samples)
    if parsed is None:
        return "fail", None
    models_gone = None
    gateway_gone = None
    for sample in parsed:
        yolo = sample["yolov8s_alive"]
        face = sample["libface_alive"]
        gateway = sample["gateway_alive"]
        supervisor = sample["supervisor_alive"]
        both_dead = (not yolo) and (not face)
        if models_gone is None:
            if not gateway:
                return "gateway_first", None
            if ((not yolo) or (not face)) and (not supervisor):
                return "supervisor_gone", None
            if both_dead and supervisor:
                models_gone = sample
            continue
        if yolo or face:
            if gateway_gone is None:
                return "fail", None
            return "model_reappeared", None
        if gateway_gone is None:
            if not gateway:
                gateway_gone = sample
            continue
        if gateway_gone["supervisor_alive"] is False:
            if sample["sample_index"] != gateway_gone["sample_index"] + 1:
                return "fail", None
            if yolo or face:
                return "model_reappeared", None
            if supervisor or gateway:
                return "fail", None
            return _finish(models_gone, gateway_gone, sample)
        if (not supervisor) and (not gateway):
            return _finish(models_gone, gateway_gone, sample)
        if supervisor and (not gateway):
            continue
        return "fail", None
    return "pending", None


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


def _pid_map_ok(value, supervisor_pid: int) -> bool:
    if not isinstance(value, dict) or set(value) != set(PID_KEYS):
        return False
    for key in PID_KEYS:
        if not is_json_int(value.get(key)) or value[key] != supervisor_pid:
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
    if not milestones_ok(
        record.get("models_gone"),
        record.get("gateway_gone"),
        record.get("after_supervisor"),
    ):
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


def _alive(index, supervisor, gateway, yolo, face):
    return {
        "sample_index": index,
        "supervisor_alive": supervisor,
        "gateway_alive": gateway,
        "yolov8s_alive": yolo,
        "libface_alive": face,
    }


def _synthetic_record(models_gone, gateway_gone, after):
    before = {
        "gateway": {
            "id": GATEWAY_ID,
            "state": "running",
            "pid": 33,
            "ready": True,
        },
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
        "models_gone": models_gone,
        "gateway_gone": gateway_gone,
        "after_supervisor": after,
    }


def order_rules_hold() -> bool:
    if not pack_ok(PACK_TEXT):
        return False
    if pack_ok(PACK_TEXT + "[pack.mobilenetv2]\n"):
        return False
    if pack_ok(PACK_TEXT.replace("worker_nums = 1", "decode_auto = true", 1)):
        return False

    later = [
        _alive(0, True, True, True, True),
        _alive(1, True, True, False, True),
        _alive(2, True, True, False, False),
        _alive(3, True, True, False, False),
        _alive(4, True, False, False, False),
        _alive(5, True, False, False, False),
        _alive(6, False, False, False, False),
    ]
    outcome, miles = classify_order(later)
    if outcome != "pass" or miles is None:
        return False
    if [item["sample_index"] for item in miles] != [2, 4, 6]:
        return False
    if not judge_record(_synthetic_record(*miles)):
        return False

    same = [
        _alive(0, True, True, True, True),
        _alive(1, True, True, False, False),
        _alive(2, False, False, False, False),
        _alive(3, False, False, False, False),
    ]
    outcome, miles = classify_order(same)
    if outcome != "pass" or miles is None:
        return False
    if [item["sample_index"] for item in miles] != [1, 2, 3]:
        return False

    if classify_order([_alive(0, True, True, True, True)])[0] != "pending":
        return False
    gateway_first = [
        _alive(0, True, True, True, True),
        _alive(1, True, False, True, True),
    ]
    if classify_order(gateway_first)[0] != "gateway_first":
        return False
    gateway_before_one = [
        _alive(0, True, True, True, True),
        _alive(1, True, True, False, True),
        _alive(2, True, False, False, True),
    ]
    if classify_order(gateway_before_one)[0] != "gateway_first":
        return False
    supervisor_gone = [
        _alive(0, True, True, True, True),
        _alive(1, False, True, True, True),
        _alive(2, False, True, False, False),
    ]
    if classify_order(supervisor_gone)[0] != "supervisor_gone":
        return False
    supervisor_same = [
        _alive(0, True, True, True, True),
        _alive(1, False, True, False, True),
    ]
    if classify_order(supervisor_same)[0] != "supervisor_gone":
        return False
    reappear = [
        _alive(0, True, True, False, False),
        _alive(1, True, False, False, False),
        _alive(2, True, False, True, False),
    ]
    if classify_order(reappear)[0] != "model_reappeared":
        return False
    reappear_next = [
        _alive(0, True, True, False, False),
        _alive(1, False, False, False, False),
        _alive(2, False, False, True, False),
    ]
    if classify_order(reappear_next)[0] != "model_reappeared":
        return False
    reappear_then_die = [
        _alive(0, True, True, False, False),
        _alive(1, True, False, False, False),
        _alive(2, True, False, True, False),
        _alive(3, True, False, False, False),
        _alive(4, False, False, False, False),
    ]
    if classify_order(reappear_then_die)[0] != "model_reappeared":
        return False

    dirty = _alive(0, True, True, False, False)
    dirty["pid"] = 11
    if classify_order([dirty])[0] != "fail":
        return False

    good = _synthetic_record(*classify_order(later)[1])
    bad_supervisor = copy.deepcopy(good)
    bad_supervisor["models_gone"] = _copy_sample(bad_supervisor["models_gone"])
    bad_supervisor["models_gone"]["supervisor_alive"] = False
    bad_gateway = copy.deepcopy(good)
    bad_gateway["gateway_gone"] = _copy_sample(bad_gateway["gateway_gone"])
    bad_gateway["gateway_gone"]["yolov8s_alive"] = True
    bad_after = copy.deepcopy(good)
    bad_after["after_supervisor"] = _copy_sample(bad_after["after_supervisor"])
    bad_after["after_supervisor"]["libface_alive"] = True
    bad_signal = copy.deepcopy(good)
    bad_signal["signal"] = "SIGINT"
    bad_pid = copy.deepcopy(good)
    bad_pid["signal_pid"] = bad_pid["supervisor_pid"] + 1
    bad_ppid = copy.deepcopy(good)
    bad_ppid["ppid"] = {"yolov8s": 100, "libface": 100, "gateway": 101}
    bad_ready = copy.deepcopy(good)
    bad_ready["before"]["servers"][0]["ready"] = False
    bad_word = copy.deepcopy(good)
    bad_word["before"] = dict(good["before"])
    bad_word["before"]["note"] = "systemctl"
    with_pid = copy.deepcopy(good)
    with_pid["models_gone"] = dict(good["models_gone"])
    with_pid["models_gone"]["pid"] = 11
    if any(
        judge_record(item)
        for item in (
            bad_supervisor,
            bad_gateway,
            bad_after,
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


def close_hold(state: dict) -> None:
    sock = state.get("hold")
    state["hold"] = None
    if sock is None:
        return
    try:
        sock.close()
    except OSError:
        pass


def _socket_inodes(pid: int):
    inodes = set()
    base = Path("/proc") / str(pid) / "fd"
    try:
        names = list(base.iterdir())
    except OSError:
        return inodes
    for item in names:
        try:
            target = os.readlink(item)
        except OSError:
            continue
        if target.startswith("socket:[") and target.endswith("]"):
            inodes.add(target[len("socket:[") : -1])
    return inodes


def gateway_has_peer(pid: int, listen_port: int, client_port: int) -> bool:
    """True when pid owns the established socket for this client port."""
    inodes = _socket_inodes(pid)
    if not inodes:
        return False
    listen_hex = "%04X" % listen_port
    client_hex = "%04X" % client_port
    try:
        lines = Path("/proc/net/tcp").read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 10 or parts[3] != "01" or parts[9] not in inodes:
            continue
        local, remote = parts[1], parts[2]
        if local.endswith(":" + listen_hex) and remote.endswith(":" + client_hex):
            return True
    return False


def open_gateway_hold(gateway_pid: int):
    """Accepted unread TCP session. No request bytes and no signal.

    Idle gateway stop() returns in a few milliseconds, inside one 50 ms
    sample. stop() waits while this session is open, then returns after
    the session is closed.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(2)
        sock.connect(("127.0.0.1", GATEWAY_PORT))
        client_port = sock.getsockname()[1]
        sock.settimeout(None)
    except OSError:
        sock.close()
        return None
    seen = False
    for _ in range(20):
        if gateway_has_peer(gateway_pid, GATEWAY_PORT, client_port):
            seen = True
            break
        time.sleep(0.01)
    if not seen:
        sock.close()
        return None
    return sock


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


def signal_supervisor(pid: int) -> bool:
    # This pid only. Not the process group and not a child pid.
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    return True


def collect_samples(state, proc, supervisor_pid, yolo_pid, face_pid, gateway_pid, deadline):
    samples = []
    index = 0
    next_tick = time.monotonic()
    outcome = "pending"
    milestones = None
    while time.monotonic() < deadline:
        proc.poll()
        sample = {
            "sample_index": index,
            "supervisor_alive": pid_alive(supervisor_pid),
            "gateway_alive": pid_alive(gateway_pid),
            "yolov8s_alive": pid_alive(yolo_pid),
            "libface_alive": pid_alive(face_pid),
        }
        samples.append(sample)
        if (
            state.get("hold") is not None
            and (not sample["yolov8s_alive"])
            and (not sample["libface_alive"])
            and sample["gateway_alive"]
            and sample["supervisor_alive"]
        ):
            close_hold(state)
        outcome, milestones = classify_order(samples)
        if outcome != "pending":
            break
        index += 1
        next_tick += SAMPLE_INTERVAL_S
        delay = next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
    return outcome, milestones, samples


def _run_live(state: dict) -> int:
    try:
        unit_text = SERVICE_PATH.read_text(encoding="utf-8")
    except OSError:
        return _fail("unit file unreadable\n")
    if not unit_ok(unit_text):
        return _fail("unit file does not match the stop contract\n")
    if not pack_ok(PACK_TEXT):
        return _fail("temporary pack text does not match\n")
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

    log_fp = open(Path(tmpdir) / "supervisor.log", "wb")
    state["log_fp"] = log_fp
    proc = subprocess.Popen(
        [str(binary)],
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
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return _fail("supervisor exited before both models were ready\n")
        doc = fetch_status(tokens["MORTRED_API_TOKEN"])
        if doc is not None and trio_ready(doc) is not None:
            before = doc
            break
        time.sleep(0.2)
    if before is None:
        return _fail("yolov8s and libface were not both ready\n")

    found = trio_ready(before)
    yolo, face, gateway = found
    yolo_pid = yolo["pid"]
    face_pid = face["pid"]
    gateway_pid = gateway["pid"]
    supervisor_pid = proc.pid
    if not is_json_int(supervisor_pid) or supervisor_pid <= 0:
        return _fail("supervisor pid is not a positive integer\n")
    ppid = {}
    for name, pid in (
        ("yolov8s", yolo_pid),
        ("libface", face_pid),
        ("gateway", gateway_pid),
    ):
        parent = read_ppid(pid)
        if parent != supervisor_pid:
            return _fail("child ppid does not match supervisor\n")
        ppid[name] = parent

    hold = open_gateway_hold(gateway_pid)
    if hold is None:
        return _fail("gateway did not accept an open session\n")
    state["hold"] = hold

    if not signal_supervisor(supervisor_pid):
        return _fail("could not signal supervisor\n")
    state["signaled"] = True
    signal_started = time.monotonic()
    stop_deadline = signal_started + STOP_WAIT_S
    outcome, milestones, samples = collect_samples(
        state, proc, supervisor_pid, yolo_pid, face_pid, gateway_pid, stop_deadline
    )
    if outcome != "pass" or milestones is None:
        tail = samples[-40:]
        lines = [
            "%d %d %d %d %d"
            % (
                item["sample_index"],
                int(item["supervisor_alive"]),
                int(item["gateway_alive"]),
                int(item["yolov8s_alive"]),
                int(item["libface_alive"]),
            )
            for item in tail
        ]
        sys.stderr.buffer.write(
            ("stop order outcome %s\n%s\n" % (outcome, "\n".join(lines))).encode("utf-8")
        )
        return _fail("stop order did not pass\n")
    if proc.poll() is None:
        remaining = stop_deadline - time.monotonic()
        if remaining <= 0:
            return _fail("supervisor did not exit\n")
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            return _fail("supervisor did not exit\n")

    models_gone, gateway_gone, after = milestones
    record = {
        "signal": "SIGTERM",
        "signal_pid": supervisor_pid,
        "supervisor_pid": supervisor_pid,
        "before": before,
        "pack": pack_text,
        "pids": {
            "yolov8s": yolo_pid,
            "libface": face_pid,
            "gateway": gateway_pid,
        },
        "ppid": {
            "yolov8s": ppid["yolov8s"],
            "libface": ppid["libface"],
            "gateway": ppid["gateway"],
        },
        "models_gone": models_gone,
        "gateway_gone": gateway_gone,
        "after_supervisor": after,
    }
    if not judge_record(record):
        return _fail("stop order record did not pass\n")
    if not write_record(record, tokens):
        return _fail("refusing to write stop order record\n")
    return 0


def stop_process(state: dict) -> bool:
    proc = state.get("proc")
    if proc is None:
        return True
    if proc.poll() is not None:
        return True
    if not state.get("signaled"):
        if not signal_supervisor(proc.pid):
            return False
        state["signaled"] = True
    try:
        proc.wait(timeout=STOP_WAIT_S)
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
        "hold": None,
    }
    code = 1
    try:
        code = _run_live(state)
    except Exception:
        sys.stderr.buffer.write(b"stop order run failed\n")
        code = 1
    finally:
        close_hold(state)
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
