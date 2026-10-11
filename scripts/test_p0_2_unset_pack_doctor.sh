#!/usr/bin/env bash
# P0-2: unset MORTRED_PACK product-engine gate (acceptance 1–5) and the
# already-set pack next-line. Moves or plants placeholders only. Does not call trtexec.
#
# Usage (repo root):
#   env -u MORTRED_PACK bash scripts/test_p0_2_unset_pack_doctor.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
unset MORTRED_PACK || true

Y8S_ENGINE="weights/object_detection/yolov8/yolov8s.engine.static_bs1.fp16"
M3D_ENGINE="weights/mono_depth_estimation/metric3d/metric3d_750k_512x1088.engine.static_bs1.fp16"
MV2_ENGINE="weights/classification/mobilenetv2/mobilenetv2.engine.static_bs1.fp32"
FAILS=0
STASH="$ROOT/weights/.p0-2-engine-stash-$$"
RESTORED=0
PLACED=0
PAIRS=""
DOUT=""
DRC=0

pass() { echo "  [PASS] $*"; }
fail() { echo "  [FAIL] $*"; FAILS=$((FAILS + 1)); }

restore_engines() {
    [ "$RESTORED" -eq 1 ] && return 0
    RESTORED=1
    if [ -n "$PAIRS" ]; then
        local id eng dest
        while IFS=$'\t' read -r id eng || [ -n "${id:-}" ]; do
            eng="${eng%$'\r'}"
            [ -z "$eng" ] && continue
            dest="$STASH/$eng"
            if [ -f "$dest" ]; then
                rm -f "$ROOT/$eng"
                mkdir -p "$ROOT/$(dirname "$eng")"
                mv "$dest" "$ROOT/$eng"
            elif [ "$PLACED" -eq 1 ]; then
                rm -f "$ROOT/$eng"
            fi
        done <<<"$PAIRS"
    fi
    rm -rf "$STASH"
}
mkdir -p "$STASH"
trap restore_engines EXIT

PAIRS="$(python3 - <<'PY'
import json
import sys
from pathlib import Path

frozen = [
    "mobilenetv2", "resnet", "densenet", "yolov5l", "yolov6s", "nanodet_1x5",
    "nanodet_416", "yolov7", "yolov7x", "yolov8n", "yolov8s", "yolov8l", "yolov8x",
    "libface", "centerface", "bisenetv2", "pphuman_mobile", "pphuman_lite",
    "pphuman_server", "hrnet", "msocrnet", "dbnet", "attentive_gan", "enlightengan",
    "realesrgan", "superpoint", "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14",
    "modnet", "ppmatting_512", "ppmatting_1024", "ppmatting_resnet34", "ppmatting_v2",
    "depth_vits14", "depth_vitb14", "depth_vitl14", "metric3d_512", "metric3d_1088",
]
named = {
    "yolov8s": "weights/object_detection/yolov8/yolov8s.engine.static_bs1.fp16",
    "metric3d_512": "weights/mono_depth_estimation/metric3d/metric3d_750k_512x1088.engine.static_bs1.fp16",
    "mobilenetv2": "weights/classification/mobilenetv2/mobilenetv2.engine.static_bs1.fp32",
}
data = json.loads(Path("conf/product_static_bs1.json").read_text(encoding="utf-8"))
rows = data["models"]
ids = [row["id"] for row in rows]
if len(rows) != 39 or set(ids) != set(frozen):
    sys.stderr.write("product table is not the frozen 39 ids\n")
    sys.exit(1)
by = {row["id"]: row["engine"] for row in rows}
for cid, engine in named.items():
    if by.get(cid) != engine:
        sys.stderr.write("%s engine drifted: %s\n" % (cid, by.get(cid)))
        sys.exit(1)
if "yolov8n" not in by:
    sys.stderr.write("yolov8n missing from product table\n")
    sys.exit(1)
for row in rows:
    sys.stdout.write("%s\t%s\n" % (row["id"], row["engine"]))
PY
)"

stash_originals() {
    local id eng dest
    while IFS=$'\t' read -r id eng || [ -n "${id:-}" ]; do
        eng="${eng%$'\r'}"
        [ -z "$eng" ] && continue
        if [ -e "$ROOT/$eng" ]; then
            dest="$STASH/$eng"
            mkdir -p "$(dirname "$dest")"
            mv "$ROOT/$eng" "$dest"
        fi
    done <<<"$PAIRS"
}

place_nonempty() {
    local id eng
    while IFS=$'\t' read -r id eng || [ -n "${id:-}" ]; do
        eng="${eng%$'\r'}"
        [ -z "$eng" ] && continue
        mkdir -p "$ROOT/$(dirname "$eng")"
        printf 'p0-2\n' >"$ROOT/$eng"
    done <<<"$PAIRS"
    PLACED=1
}

run_doctor() {
    local pack="${1:-}"
    set +e
    if [ -n "$pack" ]; then
        DOUT="$(MORTRED_PACK="$pack" bash "$ROOT/scripts/mortredctl_doctor.sh" --strict 2>&1)"
    else
        DOUT="$(env -u MORTRED_PACK bash "$ROOT/scripts/mortredctl_doctor.sh" --strict 2>&1)"
    fi
    DRC=$?
    set -e
    DOUT="${DOUT//$'\r'/}"
}

next_id_count() {
    printf '%s\n' "$DOUT" | grep -c -E '^next: mortredctl prepare --id ' || true
}

assert_before_live() {
    local label="$1"
    if printf '%s\n' "$DOUT" | grep -E 'supervisor:|live checks skipped' >/dev/null; then
        fail "$label reached live checks"
        return 1
    fi
    return 0
}

assert_only_id() {
    local id="$1" eng="$2" label="$3" n
    if [ "$DRC" -ne 1 ]; then
        fail "$label exit=$DRC want 1"
        return
    fi
    if ! printf '%s\n' "$DOUT" | grep -F "$eng" >/dev/null; then
        fail "$label missing engine path $eng"
        return
    fi
    if ! printf '%s\n' "$DOUT" | grep -qx "next: mortredctl prepare --id $id"; then
        fail "$label missing next line for $id"
        return
    fi
    n="$(next_id_count)"
    if [ "$n" -ne 1 ]; then
        fail "$label prepare --id count=$n want 1"
        return
    fi
    assert_before_live "$label" || return 0
    pass "$label"
}

echo "== P0-2 unset-pack doctor =="
stash_originals

echo "-- AC1 yolov8s missing --"
place_nonempty
rm -f "$ROOT/$Y8S_ENGINE"
run_doctor
assert_only_id "yolov8s" "$Y8S_ENGINE" "AC1 yolov8s"

echo "-- AC2 metric3d_512 missing --"
place_nonempty
rm -f "$ROOT/$M3D_ENGINE"
run_doctor
assert_only_id "metric3d_512" "$M3D_ENGINE" "AC2 metric3d_512"

echo "-- AC5 yolov8s empty --"
place_nonempty
: >"$ROOT/$Y8S_ENGINE"
run_doctor
assert_only_id "yolov8s" "$Y8S_ENGINE" "AC5 yolov8s empty"

echo "-- AC3 all 39 missing --"
place_nonempty
while IFS=$'\t' read -r id eng || [ -n "${id:-}" ]; do
    eng="${eng%$'\r'}"
    [ -z "$eng" ] && continue
    rm -f "$ROOT/$eng"
done <<<"$PAIRS"
run_doctor
ac3_ok=1
if [ "$DRC" -ne 1 ]; then
    fail "AC3 exit=$DRC want 1"
    ac3_ok=0
fi
n_all="$(next_id_count)"
if [ "$n_all" -ne 39 ]; then
    fail "AC3 prepare --id count=$n_all want 39"
    ac3_ok=0
fi
while IFS=$'\t' read -r id eng || [ -n "${id:-}" ]; do
    id="${id%$'\r'}"
    eng="${eng%$'\r'}"
    [ -z "$id" ] && continue
    if ! printf '%s\n' "$DOUT" | grep -F "$eng" >/dev/null; then
        fail "AC3 missing path $eng"
        ac3_ok=0
    fi
    if ! printf '%s\n' "$DOUT" | grep -qx "next: mortredctl prepare --id $id"; then
        fail "AC3 missing next for $id"
        ac3_ok=0
    fi
done <<<"$PAIRS"
if [ "$ac3_ok" -eq 1 ]; then
    if assert_before_live "AC3"; then
        pass "AC3 all 39"
    fi
fi

echo "-- AC4 each id alone --"
seen_yolov8n=0
while IFS=$'\t' read -r id eng || [ -n "${id:-}" ]; do
    id="${id%$'\r'}"
    eng="${eng%$'\r'}"
    [ -z "$id" ] && continue
    place_nonempty
    rm -f "$ROOT/$eng"
    run_doctor
    if [ "$id" = "mobilenetv2" ] && [ "$eng" != "$MV2_ENGINE" ]; then
        fail "AC4 mobilenetv2 path is $eng"
        continue
    fi
    assert_only_id "$id" "$eng" "AC4 $id"
    if [ "$id" = "yolov8n" ]; then
        seen_yolov8n=1
    fi
done <<<"$PAIRS"
if [ "$seen_yolov8n" -ne 1 ]; then
    fail "AC4 yolov8n did not appear"
else
    pass "AC4 yolov8n appeared"
fi

assert_pack_next() {
    local pack="$1" id="$2" eng="$3" label="$4" n
    place_nonempty
    rm -f "$ROOT/$eng"
    run_doctor "$pack"
    if [ "$DRC" -ne 1 ]; then
        fail "$label exit=$DRC want 1"
        return
    fi
    if ! printf '%s\n' "$DOUT" | grep -F "$eng" >/dev/null; then
        fail "$label missing engine path $eng"
        return
    fi
    if ! printf '%s\n' "$DOUT" | grep -qx "next: mortredctl prepare --pack $pack"; then
        fail "$label missing prepare --pack $pack"
        return
    fi
    n="$(next_id_count)"
    if [ "$n" -ne 0 ]; then
        fail "$label unexpectedly printed prepare --id ($n)"
        return
    fi
    assert_before_live "$label" || return 0
    pass "$label ($id)"
}

echo "-- MORTRED_PACK yolov8.toml --"
assert_pack_next "conf/packs/yolov8.toml" "yolov8s" "$Y8S_ENGINE" "pack yolov8"

echo "-- MORTRED_PACK metric3d.toml --"
assert_pack_next "conf/packs/metric3d.toml" "metric3d_512" "$M3D_ENGINE" "pack metric3d"

restore_engines
trap - EXIT

if [ "$FAILS" -ne 0 ]; then
    echo "[FAIL] P0-2 unset-pack doctor: $FAILS check(s) failed"
    exit 1
fi
echo "[ok] P0-2 unset-pack doctor passed"
exit 0
