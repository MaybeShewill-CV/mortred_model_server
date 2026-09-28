#!/usr/bin/env bash
# P0-1 black-box regression for product TRT path + init/doctor fail-closed (AC1–AC7).
# AC8 (prepare closed loop) requires GPU/trtexec/onnx — skipped here when unavailable.
#
# Usage (repo root):
#   bash scripts/test_p0_1_engine_fail_closed.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=python3
PACK="$ROOT/conf/packs/yolov8.toml"
AC1_ENGINE="weights/object_detection/yolov8/yolov8s.engine.static_bs1.fp16"
FAILS=0

pass() { echo "  [PASS] $*"; }
fail() { echo "  [FAIL] $*"; FAILS=$((FAILS + 1)); }
skip() { echo "  [SKIP] $*"; }

norm_path() {
  local p="$1"
  p="${p#./}"
  while [[ "$p" == ../* ]]; do p="${p#../}"; done
  printf '%s' "$p"
}

echo "== P0-1 engine fail-closed (AC1–AC7) =="

# ---- AC1: single product engine path ----
echo "-- AC1 single path --"
AC1_OUT="$($PY - <<'PY'
import json, re
from pathlib import Path
root = Path(".")
ps = json.loads((root / "conf/product_static_bs1.json").read_text())
engine_ps = None
models = ps["models"]
if isinstance(models, list):
    for m in models:
        if str(m.get("id", "")).lower() == "yolov8s":
            engine_ps = m["engine"]
            break
else:
    for mid, m in models.items():
        if str(mid).lower() == "yolov8s":
            engine_ps = m["engine"]
            break
sa = json.loads((root / "conf/trt_engines.set_a.json").read_text())
sa_e = next(e["engine"] for e in sa["engines"] if e.get("model") == "yolov8s")
osj = json.loads((root / "conf/onnx_sources.json").read_text())
rp = next(m["runtime_path"] for m in osj["models"] if m.get("id") == "YOLOV8")
toml = (root / "conf/model/object_detection/yolov8/yolov8s.toml").read_text()
toml_p = re.search(r'model_file_path\s*=\s*"([^"]+)"', toml).group(1)

def norm(p: str) -> str:
    p = p.replace("\\", "/")
    while p.startswith("../"):
        p = p[3:]
    if p.startswith("./"):
        p = p[2:]
    return p

vals = [norm(engine_ps), norm(toml_p), norm(sa_e), norm(rp)]
ok = len(set(vals)) == 1 and vals[0].endswith(".engine.static_bs1.fp16")
ok = ok and "fp16in" not in vals[0] and not vals[0].endswith("yolov8s.engine")
print(vals[0])
print("OK" if ok else "BAD")
PY
)"
AC1_PATH="$(printf '%s\n' "$AC1_OUT" | head -1)"
AC1_OK="$(printf '%s\n' "$AC1_OUT" | tail -1)"
if [ "$AC1_OK" = "OK" ] && [ "$AC1_PATH" = "$AC1_ENGINE" ]; then
  pass "AC1 paths agree on $AC1_ENGINE"
else
  fail "AC1 path mismatch: $AC1_OUT"
fi
$PY scripts/product_static_bs1.py --check >/dev/null
$PY scripts/product_static_bs1.py --check-pack "$PACK" >/dev/null
pass "AC1 product_static_bs1 --check / --check-pack"

# ---- AC2: prepare targets AC1 path (manifest + filters; no trtexec required) ----
echo "-- AC2 prepare target path --"
MANIFEST="$(mktemp)"
$PY scripts/product_static_bs1.py --write-manifest "$MANIFEST" >/dev/null
FILTERS="$($PY scripts/product_static_bs1.py --project-root "$ROOT" --pack "$PACK" --convert-filters)"
LIST="$($PY scripts/pack_trt.py --project-root "$ROOT" --pack "$PACK" --list)"
rm -f "$MANIFEST"
if printf '%s\n' "$FILTERS" | grep -qx "$AC1_ENGINE" \
  && printf '%s\n' "$LIST" | grep -F "$AC1_ENGINE" >/dev/null \
  && ! printf '%s\n' "$FILTERS$LIST" | grep -E 'yolov8s\.engine([^.]|$)|fp16in' >/dev/null; then
  pass "AC2 prepare targets $AC1_ENGINE (not legacy bare/fp16in)"
else
  fail "AC2 prepare target drift: filters='$FILTERS' list='$LIST'"
fi

# ---- AC3: doctor fail-closed on missing/empty engine ----
echo "-- AC3 doctor missing engine --"
rm -f "$ROOT/$AC1_ENGINE"
set +e
DOC_OUT="$(MORTRED_PACK="$PACK" bash scripts/mortredctl_doctor.sh 2>&1)"
DOC_RC=$?
set -e
if [ "$DOC_RC" -ne 0 ] && printf '%s\n' "$DOC_OUT" | grep -F "$AC1_ENGINE" >/dev/null; then
  pass "AC3 doctor exit=$DOC_RC names missing engine"
else
  fail "AC3 doctor exit=$DOC_RC output did not name $AC1_ENGINE"
fi

# ---- AC4: doctor fail-closed on missing stamp (engine present) ----
echo "-- AC4 doctor missing stamp --"
mkdir -p "$(dirname "$ROOT/$AC1_ENGINE")"
printf 'dummy\n' > "$ROOT/$AC1_ENGINE"
set +e
DOC_OUT="$(MORTRED_PACK="$PACK" bash scripts/mortredctl_doctor.sh 2>&1)"
DOC_RC=$?
set -e
rm -f "$ROOT/$AC1_ENGINE"
if [ "$DOC_RC" -ne 0 ] && printf '%s\n' "$DOC_OUT" | grep -E 'gpu_mem_mib|calibrate' >/dev/null; then
  pass "AC4 doctor exit=$DOC_RC points at stamp/calibrate"
else
  fail "AC4 doctor exit=$DOC_RC missing stamp/calibrate signal"
fi

# ---- AC5 / AC6 / AC7: init with fetch_weights stubs ----
STUB_DIR="$(mktemp -d)"
cleanup_init_stubs() {
  if [ -f "$ROOT/scripts/fetch_weights.py.bak_p01qa" ]; then
    mv "$ROOT/scripts/fetch_weights.py.bak_p01qa" "$ROOT/scripts/fetch_weights.py"
  fi
  if [ -f "$ROOT/scripts/verify_deployment.sh.bak_p01qa" ]; then
    mv "$ROOT/scripts/verify_deployment.sh.bak_p01qa" "$ROOT/scripts/verify_deployment.sh"
  fi
  rm -rf "$STUB_DIR"
}
trap cleanup_init_stubs EXIT
cp "$ROOT/scripts/fetch_weights.py" "$ROOT/scripts/fetch_weights.py.bak_p01qa"

echo "-- AC5 init weight failure --"
cat > "$ROOT/scripts/fetch_weights.py" <<'EOF'
#!/usr/bin/env python3
import sys
print("fetch_weights stub: simulated failure", file=sys.stderr)
sys.exit(2)
EOF
set +e
INIT_OUT="$(bash scripts/mortredctl_init.sh --profile gpu 2>&1)"
INIT_RC=$?
set -e
if [ "$INIT_RC" -ne 0 ] && ! printf '%s\n' "$INIT_OUT" | grep -Eiq 'init done'; then
  pass "AC5 init exit=$INIT_RC, no success 'init done'"
else
  fail "AC5 init exit=$INIT_RC or printed init done"
fi

echo "-- AC6 init gpu missing engine --"
cat > "$ROOT/scripts/fetch_weights.py" <<'EOF'
#!/usr/bin/env python3
print("fetch_weights stub: ok")
raise SystemExit(0)
EOF
rm -f "$ROOT/$AC1_ENGINE"
set +e
INIT_OUT="$(MORTRED_PACK="$PACK" bash scripts/mortredctl_init.sh --profile gpu 2>&1)"
INIT_RC=$?
set -e
if [ "$INIT_RC" -ne 0 ] && printf '%s\n' "$INIT_OUT" | grep -F "$AC1_ENGINE" >/dev/null \
  && ! printf '%s\n' "$INIT_OUT" | grep -Eiq 'init done'; then
  pass "AC6 init exit=$INIT_RC names missing engine"
else
  fail "AC6 init exit=$INIT_RC / missing engine name / init done present"
fi

echo "-- AC7 init cpu no TRT gate --"
if [ -x "$ROOT/scripts/verify_deployment.sh" ]; then
  cp "$ROOT/scripts/verify_deployment.sh" "$ROOT/scripts/verify_deployment.sh.bak_p01qa"
  cat > "$ROOT/scripts/verify_deployment.sh" <<'EOF'
#!/usr/bin/env bash
echo "verify stub ok"
exit 0
EOF
  chmod +x "$ROOT/scripts/verify_deployment.sh"
fi
set +e
INIT_OUT="$(bash scripts/mortredctl_init.sh --profile cpu 2>&1)"
INIT_RC=$?
set -e
if [ "$INIT_RC" -eq 0 ] && ! printf '%s\n' "$INIT_OUT" | grep -Eiq 'pack TensorRT engine missing|TRT engines not ready'; then
  pass "AC7 init cpu exit=0 without TRT engine requirement"
else
  fail "AC7 init cpu exit=$INIT_RC unexpectedly gated on TRT"
fi

# ---- AC8: only announce skip/block when GPU toolchain missing ----
echo "-- AC8 prepare closed loop --"
if command -v trtexec >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 \
  && [ -f "$ROOT/weights/object_detection/yolov8/yolov8s.static_bs1.onnx" ]; then
  skip "AC8 runnable here — run manually: mortredctl prepare --pack conf/packs/yolov8.toml --skip-ready"
else
  skip "AC8 blocked (need GPU + trtexec + onnx); not asserted in this script"
fi

cleanup_init_stubs
trap - EXIT

# white-box: product static_bs1 must not be forced into legacy trt_engines.json
echo "-- whitebox consistency legacy gate --"
set +e
CONS_OUT="$($PY scripts/check_consistency.py 2>&1)"
CONS_RC=$?
set -e
if printf '%s\n' "$CONS_OUT" | grep -E 'static_bs1.*trt_engines\.json|trt_engines\.json.*static_bs1' >/dev/null; then
  fail "check_consistency still binds product static_bs1 to legacy trt_engines.json"
else
  pass "check_consistency exit=$CONS_RC with no product-static↔legacy-json bind error"
fi

if [ "$FAILS" -ne 0 ]; then
  echo "[FAIL] P0-1 regression: $FAILS check(s) failed"
  exit 1
fi
echo "[ok] P0-1 AC1–AC7 regression passed"
exit 0
