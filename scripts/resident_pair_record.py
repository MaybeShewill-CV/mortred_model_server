#!/usr/bin/env python3
"""Record one overlapping infer while yolov8s and libface are both ready.

The live run and --check call the same judge(). --check does not start
processes and does not read, write, or delete the live json.
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORD_PATH = ROOT / "logs" / "resident_pair_yolov8s_libface.json"

YOLO_ID = "yolov8s"
FACE_ID = "libface"
YOLO_PATH = "/v1/models/yolov8s/infer"
FACE_PATH = "/v1/models/libface/infer"
YOLO_IMAGE = ROOT / "demo_data/model_test_input/object_detection/bus.jpg"
FACE_IMAGE = ROOT / "demo_data/model_test_input/object_detection/face_wo_mask.jpg"

STATUS_URL_PORT = 8787
GATEWAY_PORT = 8080
PORTS = (8787, 8080, 9051, 9056)
# Existing autostart budgets. Do not extend them.
READY_BUDGET_S = 15 + 60
INFER_TIMEOUT_S = 60
STOP_WAIT_S = 30

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

CHECK_STDOUT = (
    "时间重叠且两边成功为通过\n"
    "时间不重叠为不通过\n"
    "before 里一边 ready 不为 true 为不通过\n"
    "after 的 pid 与 before 不一致为不通过\n"
    "任一条 HTTP 不是 200、响应根上 status 不是 0 或 results[0].status 不是 0 为不通过\n"
    "pack 全文出现第三个 [pack. 节为不通过\n"
)

_SEED_LINES = (
    "worker_nums = 1",
    "gpu_mem_mib = 1800",
    "gpu_mem_at_workers = 1",
    'gpu_mem_source = "seed"',
)


def is_json_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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
    """yolov8s and libface from servers; gateway only from key gateway."""
    if not isinstance(status, dict):
        return None
    yolo = one_server(status, YOLO_ID)
    face = one_server(status, FACE_ID)
    gateway = status.get("gateway")
    if yolo is None or face is None:
        return None
    if not isinstance(gateway, dict) or gateway.get("id") != "__gateway":
        return None
    return yolo, face, gateway


def place_before(obj: dict) -> bool:
    if obj.get("ready") is not True:
        return False
    if obj.get("state") != "running":
        return False
    pid = obj.get("pid")
    if not is_json_int(pid) or pid <= 0:
        return False
    # JSON integer only. Not compared with before_sample, start, or end.
    if not is_json_int(obj.get("started_at_ms")):
        return False
    return True


def place_after(before_obj: dict, after_obj: dict) -> bool:
    if after_obj.get("ready") is not True:
        return False
    if after_obj.get("state") != "running":
        return False
    before_pid = before_obj.get("pid")
    after_pid = after_obj.get("pid")
    if not is_json_int(before_pid) or not is_json_int(after_pid):
        return False
    return after_pid == before_pid


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


def response_ok(item: dict) -> bool:
    if not is_json_int(item.get("http_status")) or item.get("http_status") != 200:
        return False
    response = item.get("response")
    if not isinstance(response, dict):
        return False
    status = response.get("status")
    if not is_json_int(status) or status != 0:
        return False
    results = response.get("results")
    if not isinstance(results, list) or len(results) != 1:
        return False
    first = results[0]
    if not isinstance(first, dict):
        return False
    item_status = first.get("status")
    return is_json_int(item_status) and item_status == 0


def requests_by_path(record: dict):
    items = record.get("requests")
    if not isinstance(items, list) or len(items) != 2:
        return None
    found = {}
    for item in items:
        if not isinstance(item, dict):
            return None
        path = item.get("path")
        if not isinstance(path, str) or path in found:
            return None
        if not is_json_int(item.get("start")) or not is_json_int(item.get("end")):
            return None
        if not isinstance(item.get("response"), dict):
            return None
        if not is_json_int(item.get("http_status")):
            return None
        found[path] = item
    if set(found) != {YOLO_PATH, FACE_PATH}:
        return None
    return found


def judge(record) -> bool:
    """Acceptance 3–7. Four integer comparisons, not five."""
    if not isinstance(record, dict):
        return False
    before_sample = record.get("before_sample")
    after_sample = record.get("after_sample")
    if not is_json_int(before_sample) or not is_json_int(after_sample):
        return False
    before = trio(record.get("before"))
    after = trio(record.get("after"))
    if before is None or after is None:
        return False
    if not all(place_before(obj) for obj in before):
        return False
    if not all(place_after(src, dst) for src, dst in zip(before, after)):
        return False
    if not pack_ok(record.get("pack")):
        return False
    found = requests_by_path(record)
    if found is None:
        return False
    yolo = found[YOLO_PATH]
    face = found[FACE_PATH]
    yolo_start = yolo["start"]
    yolo_end = yolo["end"]
    face_start = face["start"]
    face_end = face["end"]
    # before_sample, two starts, two ends. after_sample is not one of them.
    if not (
        yolo_start > before_sample
        and face_start > before_sample
        and yolo_start < face_end
        and face_start < yolo_end
    ):
        return False
    return response_ok(yolo) and response_ok(face)


def _status_body(
    yolo_ready=True,
    face_ready=True,
    yolo_pid=11,
    face_pid=22,
    gateway_pid=33,
):
    def place(model_id, ready, pid):
        return {
            "id": model_id,
            "state": "running",
            "pid": pid,
            "ready": ready,
            "started_at_ms": 5000,
        }

    return {
        "gateway": place("__gateway", True, gateway_pid),
        "servers": [
            place(YOLO_ID, yolo_ready, yolo_pid),
            place(FACE_ID, face_ready, face_pid),
            {
                "id": "mobilenetv2",
                "state": "stopped",
                "pid": 0,
                "ready": False,
                "started_at_ms": 0,
            },
        ],
    }


def _response(root_status=0, item_status=0, extra=False):
    results = [{"status": item_status}]
    if extra:
        results.append({"status": 0})
    return {"status": root_status, "results": results}


def _sample(
    yolo_end=3000,
    face_start=1600,
    face_end=3100,
    yolo_ready=True,
    face_ready=True,
    after_yolo_pid=11,
    after_face_pid=22,
    after_gateway_pid=33,
    yolo_http=200,
    yolo_root=0,
    yolo_item=0,
    yolo_extra=False,
    pack=None,
):
    return {
        "before": _status_body(yolo_ready=yolo_ready, face_ready=face_ready),
        "before_sample": 1000,
        "after": _status_body(
            yolo_pid=after_yolo_pid,
            face_pid=after_face_pid,
            gateway_pid=after_gateway_pid,
        ),
        "after_sample": 4000,
        "pack": PACK_TEXT if pack is None else pack,
        "requests": [
            {
                "path": YOLO_PATH,
                "http_status": yolo_http,
                "response": _response(yolo_root, yolo_item, yolo_extra),
                "start": 1500,
                "end": yolo_end,
            },
            {
                "path": FACE_PATH,
                "http_status": 200,
                "response": _response(),
                "start": face_start,
                "end": face_end,
            },
        ],
    }


def run_check() -> int:
    # Stat only. Do not open, write, or delete the live record.
    if RECORD_PATH.exists():
        sys.stderr.write(
            "logs/resident_pair_yolov8s_libface.json already exists\n"
        )
        return 1
    samples_ok = (
        judge(_sample())
        and not judge(_sample(yolo_end=2000, face_start=2500, face_end=3200))
        and not judge(_sample(face_ready=False))
        and not judge(_sample(yolo_ready=False))
        and not judge(_sample(after_yolo_pid=99))
        and not judge(_sample(after_face_pid=99))
        and not judge(_sample(after_gateway_pid=99))
        and not judge(_sample(yolo_http=500))
        and not judge(_sample(yolo_root=68))
        and not judge(_sample(yolo_item=1))
        and not judge(_sample(pack=PACK_TEXT + "[pack.mobilenetv2]\n"))
        and not judge(_sample(yolo_extra=True))
    )
    if not samples_ok:
        sys.stderr.write("resident pair check samples did not match\n")
        return 1
    sys.stdout.write(CHECK_STDOUT)
    return 0


def _fail(message: str) -> int:
    sys.stderr.write(message)
    if not message.endswith("\n"):
        sys.stderr.write("\n")
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
    if not missing:
        return found
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
    missing = [key for key in TOKEN_KEYS if key not in found]
    if missing:
        return None
    return found


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
            "127.0.0.1", STATUS_URL_PORT, "GET", "/api/v1/status", token, None, 5
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


def models_ready(doc: dict) -> bool:
    yolo = one_server(doc, YOLO_ID)
    face = one_server(doc, FACE_ID)
    return (
        isinstance(yolo, dict)
        and yolo.get("ready") is True
        and isinstance(face, dict)
        and face.get("ready") is True
    )


def infer_once(path, body, token, barrier):
    try:
        barrier.wait(timeout=5)
    except threading.BrokenBarrierError:
        now = time.monotonic_ns()
        return {
            "path": path,
            "http_status": 0,
            "response": {},
            "start": now,
            "end": now,
        }
    start = time.monotonic_ns()
    http_status = 0
    response = {}
    try:
        status, payload = exchange(
            "127.0.0.1", GATEWAY_PORT, "POST", path, token, body, INFER_TIMEOUT_S
        )
        http_status = status
        parsed = json.loads(payload.decode("utf-8"))
        if isinstance(parsed, dict):
            response = parsed
    except (OSError, http.client.HTTPException, UnicodeError, json.JSONDecodeError):
        pass
    end = time.monotonic_ns()
    return {
        "path": path,
        "http_status": http_status,
        "response": response,
        "start": start,
        "end": end,
    }


def image_body(path: Path) -> bytes:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return json.dumps({"images": [encoded]}).encode("utf-8")


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


def write_record(record: dict, tokens: dict) -> bool:
    text = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    for value in tokens.values():
        if value and value in text:
            return False
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RECORD_PATH.with_name(RECORD_PATH.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp, RECORD_PATH)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return False
    return True


def stop_supervisor(proc) -> bool:
    if proc is None:
        return True
    if proc.poll() is not None:
        return True
    try:
        os.kill(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    try:
        proc.wait(timeout=STOP_WAIT_S)
    except subprocess.TimeoutExpired:
        return False
    return True


def remove_pack(pack_path, tmpdir) -> bool:
    ok = True
    if pack_path is not None:
        try:
            pack_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            ok = False
    if tmpdir is not None:
        try:
            os.rmdir(tmpdir)
        except FileNotFoundError:
            pass
        except OSError:
            ok = False
    return ok


def _run_live(state: dict) -> int:
    binary = find_supervisor()
    if binary is None:
        return _fail("mortred-supervisor.out not found\n")
    if not YOLO_IMAGE.is_file() or not FACE_IMAGE.is_file():
        return _fail("demo image missing\n")
    tokens = load_tokens()
    if tokens is None:
        return _fail("missing management, inference, or metrics token\n")
    busy = [port for port in PORTS if port_in_use(port)]
    if busy:
        listed = " ".join(str(port) for port in busy)
        return _fail("port already in use: %s\n" % listed)
    try:
        yolo_body = image_body(YOLO_IMAGE)
        face_body = image_body(FACE_IMAGE)
    except OSError:
        return _fail("demo image unreadable\n")

    tmpdir = tempfile.mkdtemp(prefix="mortred-resident-pair-")
    state["tmpdir"] = tmpdir
    pack_path = Path(tmpdir) / "pack.toml"
    pack_path.write_bytes(PACK_TEXT.encode("utf-8"))
    state["pack_path"] = pack_path
    pack_text = pack_path.read_text(encoding="utf-8")

    log_path = ROOT / "logs" / "resident_pair_supervisor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_path, "wb")
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
    before_sample = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return _fail("supervisor exited before both models were ready\n")
        doc = fetch_status(tokens["MORTRED_API_TOKEN"])
        if doc is not None and models_ready(doc):
            before = doc
            before_sample = time.monotonic_ns()
            break
        time.sleep(0.2)
    if before is None:
        return _fail("yolov8s and libface were not both ready\n")

    barrier = threading.Barrier(2)
    slots = {}

    def runner(key, path, body):
        slots[key] = infer_once(
            path, body, tokens["MORTRED_GATEWAY_AUTH_TOKEN"], barrier
        )

    threads = [
        threading.Thread(target=runner, args=("yolo", YOLO_PATH, yolo_body)),
        threading.Thread(target=runner, args=("face", FACE_PATH, face_body)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=INFER_TIMEOUT_S + 15)
    if any(thread.is_alive() for thread in threads) or "yolo" not in slots or "face" not in slots:
        return _fail("infer did not finish\n")

    after = None
    after_sample = None
    status_deadline = time.monotonic() + 5
    while time.monotonic() < status_deadline:
        doc = fetch_status(tokens["MORTRED_API_TOKEN"])
        if doc is not None:
            after = doc
            after_sample = time.monotonic_ns()
            break
        time.sleep(0.2)
    if after is None:
        return _fail("after status was not read\n")

    record = {
        "before": before,
        "before_sample": before_sample,
        "after": after,
        "after_sample": after_sample,
        "pack": pack_text,
        "requests": [slots["yolo"], slots["face"]],
    }
    if not judge(record):
        return _fail("resident pair record did not pass\n")
    if not write_record(record, tokens):
        return _fail("refusing to write resident pair record\n")
    return 0


def run_live() -> int:
    state = {"proc": None, "pack_path": None, "tmpdir": None, "log_fp": None}
    code = 1
    try:
        code = _run_live(state)
    except Exception:
        sys.stderr.write("resident pair run failed\n")
        code = 1
    finally:
        stopped = stop_supervisor(state["proc"])
        removed = remove_pack(state["pack_path"], state["tmpdir"])
        log_fp = state["log_fp"]
        if log_fp is not None:
            try:
                log_fp.close()
            except OSError:
                pass
        if not stopped or not removed:
            sys.stderr.write("resident pair cleanup failed\n")
    return code


def main(argv: list[str]) -> int:
    if argv == ["--check"]:
        return run_check()
    if argv == []:
        return run_live()
    sys.stderr.write("usage: python3 scripts/resident_pair_record.py [--check]\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
