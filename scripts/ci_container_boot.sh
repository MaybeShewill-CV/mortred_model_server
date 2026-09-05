#!/usr/bin/env bash
# ci_container_boot.sh - hosted-CI / local acceptance for the cpu runtime image.
#
# Builds compose --profile cpu, waits for supervisor health + gateway /healthz,
# waits until pack model MOBILENETV2 is ready, posts one gateway infer, then
# checks mortred-supervisor.out and mortred-gateway.out are still in the
# container's process table.
#
# Required env (fail-closed, must be three distinct values):
#   MORTRED_API_TOKEN  MORTRED_GATEWAY_AUTH_TOKEN  MORTRED_METRICS_TOKEN
# Optional:
#   SKIP_DOCKER_CHECK=1   skip in-image `check` (CI only; default 0)
#   WEIGHTS_DIR           host weights mount (default ./weights)
#   BOOT_TIMEOUT_S        per-wait timeout (default 180)
#
# Autostart uses demo pack id MOBILENETV2 with a CI-only device=cpu model toml
# overlay (git copies stay device=gpu). This is not a GPU / TensorRT gate.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

need_token() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "[FAIL] $name is unset" >&2
        exit 1
    fi
}

need_token MORTRED_API_TOKEN
need_token MORTRED_GATEWAY_AUTH_TOKEN
need_token MORTRED_METRICS_TOKEN

if [ "$MORTRED_API_TOKEN" = "$MORTRED_GATEWAY_AUTH_TOKEN" ] \
        || [ "$MORTRED_API_TOKEN" = "$MORTRED_METRICS_TOKEN" ] \
        || [ "$MORTRED_GATEWAY_AUTH_TOKEN" = "$MORTRED_METRICS_TOKEN" ]; then
    echo "[FAIL] API / gateway / metrics tokens must be three distinct values" >&2
    exit 1
fi

export WEIGHTS_DIR="${WEIGHTS_DIR:-$ROOT/weights}"
SKIP_DOCKER_CHECK="${SKIP_DOCKER_CHECK:-0}"
BOOT_TIMEOUT_S="${BOOT_TIMEOUT_S:-180}"
IMAGE_JPEG="$ROOT/demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG"

if [ ! -f "$IMAGE_JPEG" ]; then
    echo "[FAIL] missing demo image $IMAGE_JPEG" >&2
    exit 1
fi

echo "== fetch MOBILENETV2 weights =="
python3 "$ROOT/scripts/fetch_weights.py" --only mobilenetv2_ilsvrc2012.mnn
python3 "$ROOT/scripts/fetch_weights.py" --check --only mobilenetv2_ilsvrc2012.mnn

# Git model tomls default device=gpu (MNN_FORWARD_CUDA). Hosted CPU VMs have
# no CUDA; pack-override a cpu-device copy instead of resurrecting *_cpu_config.toml.
CI_BOOT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mortred-ci-boot.XXXXXX")"
COMPOSE=()
booted=0
cleanup() {
    local ec=$?
    if [ "$booted" = "1" ]; then
        if [ "$ec" -ne 0 ]; then
            echo "== compose logs (failure) ==" >&2
            "${COMPOSE[@]}" logs --no-color || true
        fi
        "${COMPOSE[@]}" down -v || true
    fi
    rm -rf "$CI_BOOT_DIR"
}
trap cleanup EXIT
python3 - "$ROOT" "$CI_BOOT_DIR" <<'PY'
import pathlib, re, sys
root = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
src = root / "conf/model/classification/mobilenetv2/mobilenetv2_config.toml"
text = src.read_text(encoding="utf-8")
patched, n = re.subn(r'(device\s*=\s*")gpu(")', r"\1cpu\2", text, count=1)
if n != 1:
    raise SystemExit("failed to rewrite backend.device gpu -> cpu in %s" % src)
(out / "mobilenetv2_config.toml").write_text(patched, encoding="utf-8")
(out / "ci_pack.toml").write_text(
    "[pack.MOBILENETV2]\n"
    "worker_nums = 1\n"
    'model_config = "/tmp/ci_mobilenetv2.toml"\n',
    encoding="utf-8",
)
(out / "compose.override.yml").write_text(
    "services:\n"
    "  mortred-cpu:\n"
    "    environment:\n"
    "      MORTRED_PACK: /opt/mortred/conf/packs/ci_cpu.toml\n"
    "    volumes:\n"
    "      - %s:/opt/mortred/conf/packs/ci_cpu.toml:ro\n"
    "      - %s:/tmp/ci_mobilenetv2.toml:ro\n"
    % ((out / "ci_pack.toml").as_posix(), (out / "mobilenetv2_config.toml").as_posix()),
    encoding="utf-8",
)
print("[ok] cpu-device pack overlay in %s" % out)
PY

COMPOSE=(docker compose --profile cpu
         -f "$ROOT/docker-compose.yml"
         -f "$CI_BOOT_DIR/compose.override.yml")

echo "== docker compose --profile cpu build (SKIP_DOCKER_CHECK=${SKIP_DOCKER_CHECK}) =="
"${COMPOSE[@]}" build --build-arg SKIP_DOCKER_CHECK="$SKIP_DOCKER_CHECK"
echo "== docker compose --profile cpu up =="
booted=1
"${COMPOSE[@]}" up -d --no-build

echo "== wait supervisor /api/v1/health =="
deadline=$((SECONDS + BOOT_TIMEOUT_S))
until curl -fsS --max-time 5 http://127.0.0.1:8787/api/v1/health >/dev/null; do
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "[FAIL] supervisor health timed out" >&2
        exit 1
    fi
    sleep 2
done
echo "[ok] supervisor health"

echo "== wait gateway /healthz =="
deadline=$((SECONDS + BOOT_TIMEOUT_S))
until curl -fsS --max-time 5 http://127.0.0.1:8080/healthz >/dev/null; do
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "[FAIL] gateway /healthz timed out" >&2
        exit 1
    fi
    sleep 2
done
echo "[ok] gateway /healthz"

echo "== wait MOBILENETV2 ready =="
BOOT_TIMEOUT_S="$BOOT_TIMEOUT_S" python3 - <<'PY'
import json, os, time, urllib.error, urllib.request

token = os.environ["MORTRED_API_TOKEN"]
timeout_s = int(os.environ["BOOT_TIMEOUT_S"])
deadline = time.time() + timeout_s
last = None
while time.time() < deadline:
    req = urllib.request.Request(
        "http://127.0.0.1:8787/api/v1/status",
        headers={"Authorization": "Bearer " + token},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            last = json.load(resp)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        last = {"error": str(exc)}
        time.sleep(2)
        continue
    gateway = last.get("gateway") or {}
    servers = last.get("servers") or []
    mv = next((s for s in servers if s.get("id") == "MOBILENETV2"), None)
    if gateway.get("ready") and mv and mv.get("ready") and mv.get("state") == "running":
        print("[ok] gateway + MOBILENETV2 ready")
        raise SystemExit(0)
    time.sleep(2)
print("[FAIL] timed out waiting for MOBILENETV2 ready", file=__import__("sys").stderr)
print(json.dumps(last, indent=2)[:4000], file=__import__("sys").stderr)
raise SystemExit(1)
PY

echo "== one gateway infer (MOBILENETV2) =="
python3 - "$IMAGE_JPEG" <<'PY'
import base64, json, os, pathlib, sys, urllib.error, urllib.request

image_path = pathlib.Path(sys.argv[1])
body = json.dumps({
    "images": [base64.b64encode(image_path.read_bytes()).decode()],
    "req_id": "ci-container-boot",
}).encode()
token = os.environ["MORTRED_GATEWAY_AUTH_TOKEN"]
req = urllib.request.Request(
    "http://127.0.0.1:8080/v1/models/MOBILENETV2/infer",
    data=body,
    method="POST",
    headers={
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
    },
)
try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
        status = resp.status
except urllib.error.HTTPError as exc:
    err_body = exc.read().decode("utf-8", errors="replace")
    print("[FAIL] infer HTTP %s: %s" % (exc.code, err_body[:2000]), file=sys.stderr)
    raise SystemExit(1)
payload = json.loads(raw.decode("utf-8"))
if status != 200:
    print("[FAIL] infer HTTP %s: %s" % (status, payload), file=sys.stderr)
    raise SystemExit(1)
if not isinstance(payload, dict):
    print("[FAIL] infer response is not a JSON object", file=sys.stderr)
    raise SystemExit(1)
code = payload.get("code", 0)
if code not in (0, "0"):
    print("[FAIL] infer business code %r: %s" % (code, json.dumps(payload)[:2000]), file=sys.stderr)
    raise SystemExit(1)
if payload.get("results") is None and payload.get("data") is None:
    print("[FAIL] infer missing results/data: %s" % list(payload), file=sys.stderr)
    raise SystemExit(1)
print("[ok] infer HTTP 200 code=0 keys=%s" % sorted(payload.keys()))
PY

echo "== supervisor / gateway processes still alive =="
top_out="$(docker top mortred-cpu)"
echo "$top_out"
echo "$top_out" | grep -q "mortred-supervisor.out" || {
    echo "[FAIL] mortred-supervisor.out not in docker top" >&2
    exit 1
}
echo "$top_out" | grep -q "mortred-gateway.out" || {
    echo "[FAIL] mortred-gateway.out not in docker top" >&2
    exit 1
}
echo "[ok] supervisor and gateway processes are running"
echo "== cpu compose boot passed =="
