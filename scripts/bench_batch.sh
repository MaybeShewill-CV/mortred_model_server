#!/usr/bin/env bash
# bench_batch.sh - batch-inference acceptance for every model benchmark.
#
# Runs each benchmark executable with the generic --batch N mode (provided by
# src/apps/common/benchmark_runner.h): one functional gate (every item in the
# batch must succeed) plus a short timed loop (batch/s + img/s).
#
# Usage:
#   ./scripts/bench_batch.sh                     # all models, batch=4
#   ./scripts/bench_batch.sh --only mobilenetv2  # name substring filter
#   ./scripts/bench_batch.sh --batch 8 --loops 20
#   ./scripts/bench_batch.sh --list              # show the model manifest
#
# Notes:
#   - openai_clip / bytetrack have standalone benchmark mains (no --batch
#     mode); they are listed as skipped.
#   - diffusion/sam models are slow: their default batch is 2 and timeout is
#     larger; override with --batch/--timeout if needed.
#   - model weights must be present (weights/); missing weights fail the
#     model's init and are reported as FAIL, which is the correct signal.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BATCH=4
LOOPS=""
ONLY=""
BIN_DIR=""
TIMEOUT=600
LIST=0

while [ $# -gt 0 ]; do
    case "$1" in
        --batch) BATCH="$2"; shift 2 ;;
        --loops) LOOPS="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        --bin-dir) BIN_DIR="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --list) LIST=1; shift ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "[ERROR] unknown argument: $1" >&2; exit 1 ;;
    esac
done

# benchmark name -> model config (relative to project root)
declare -A MANIFEST=(
    [densenet_benchmark]="conf/model/classification/densenet/densenet121_config.toml"
    [dinov2_benchmark]="conf/model/feature_embedding/dinov2/dinov2_vits14_config.toml"
    [mobilenetv2_benchmark]="conf/model/classification/mobilenetv2/mobilenetv2_config.toml"
    [resnet_benchmark]="conf/model/classification/resnet/resnet50_config.toml"
    [cls_cond_ddim_sampler_benchmark]="conf/model/diffusion/ddpm/cls_cond_ddim_netease-album-cover.toml"
    [ddim_sampler_benchmark]="conf/model/diffusion/ddpm/ddim_celeba-hq.toml"
    [ddpm_sampler_benchmark]="conf/model/diffusion/ddpm/ddpm_celeba-hq.toml"
    [ldm_sampler_benchmark]="conf/model/diffusion/ldm/ldm_celeba-hq.toml"
    [attentivegan_benchmark]="conf/model/enhancement/attentive_gan_derain/attentive_gan_derain_config.toml"
    [enlightengan_benchmark]="conf/model/enhancement/enlighten_gan/enlightengan.toml"
    [real_esrgan_benchmark]="conf/model/enhancement/real_esrgan/real_esrgan.toml"
    [lightglue_benchmark]="conf/model/feature_point/lightglue/lightglue_config.toml"
    [superpoint_benchmark]="conf/model/feature_point/superpoint/superpoint_config.toml"
    [modnet_benchmark]="conf/model/matting/modnet/modnet_config.toml"
    [ppmatting_benchmark]="conf/model/matting/ppmatting/ppmatting_config.toml"
    [depth_anything_benchmark]="conf/model/mono_depth_estimation/depth_anything/vit_s.toml"
    [metric3d_benchmark]="conf/model/mono_depth_estimation/metric3d/metric3d_512x1088.toml"
    [centerface_benchmark]="conf/model/object_detection/centerface/centerface_config.toml"
    [libface_benchmark]="conf/model/object_detection/libfacedetection/640x480_config.toml"
    [nanodet_benchmark]="conf/model/object_detection/nano_det/nanodet_config.toml"
    [yolov5_benchmark]="conf/model/object_detection/yolov5/yolov5_config.toml"
    [yolov6_benchmark]="conf/model/object_detection/yolov6/yolov6_config.toml"
    [yolov7_benchmark]="conf/model/object_detection/yolov7/yolov7_config.toml"
    [yolov8_benchmark]="conf/model/object_detection/yolov8/yolov8_config.toml"
    [dbnet_benchmark]="conf/model/ocr/db_text_detector/dbnet_config.toml"
    [fast_sam_benchmark]="conf/model/segment_anything/fast_sam_s_config.toml"
    [sam_amg_benchmark]="conf/model/segment_anything/mobile_sam_amg_config.toml"
    [sam_benchmark]="conf/model/segment_anything/mobile_sam_config.toml"
    [bisenetv2_benchmark]="conf/model/scene_segmentation/bisenetv2/bisenetv2_config.toml"
    [hrnet_segmentation_benchmark]="conf/model/scene_segmentation/hrnet/hrnetw48_ccd_fv_segmentation_cfg.toml"
    [msocrnet_benchmark]="conf/model/scene_segmentation/msocrnet/msocrnet_config.toml"
    [pphumanseg_benchmark]="conf/model/scene_segmentation/pphuman/pphuman_config.toml"
)
# standalone benchmark mains without the generic --batch mode
STANDALONE="openai_clip_benchmark bytetrack_benchmark"
# slow family: samplers / automask run seconds..minutes per item
SLOW="cls_cond_ddim_sampler ddim_sampler ddpm_sampler ldm_sampler sam_amg fast_sam"

if [ "$LIST" -eq 1 ]; then
    echo "batch-testable benchmarks (generic --batch mode):"
    for name in $(printf '%s\n' "${!MANIFEST[@]}" | sort); do
        slow=no
        case " $SLOW " in *" ${name%%_benchmark} "*) slow=yes ;; esac
        marker=""
        [ "$slow" = yes ] && marker="  [slow]"
        printf '  %-38s %s%s\n' "$name" "${MANIFEST[$name]}" "$marker"
    done
    echo "skipped (standalone main, no --batch mode): $STANDALONE"
    exit 0
fi

# locate the benchmark binaries (source tree _bin, install tree bin)
if [ -z "$BIN_DIR" ]; then
    for cand in "$ROOT/_bin" "$ROOT/bin"; do
        if [ -d "$cand" ]; then BIN_DIR="$cand"; break; fi
    done
fi
if [ -z "$BIN_DIR" ] || [ ! -d "$BIN_DIR" ]; then
    echo "[ERROR] no benchmark binary dir found (expected _bin or bin under $ROOT); build first or pass --bin-dir" >&2
    exit 1
fi

export LD_LIBRARY_PATH="$BIN_DIR/../_lib:$ROOT/3rd_party/libs:${LD_LIBRARY_PATH:-}"
export GLOG_logtostderr=1

PASS=0; FAIL=0; SKIP=0
FAILED_NAMES=""
for name in $(printf '%s\n' "${!MANIFEST[@]}" | sort); do
    if [ -n "$ONLY" ] && [[ "$name" != *"$ONLY"* ]]; then continue; fi
    cfg="${MANIFEST[$name]}"
    exe="$BIN_DIR/$name.out"
    slow=no
    case " $SLOW " in *" ${name%%_benchmark} "*) slow=yes ;; esac
    batch=$BATCH
    tmo=$TIMEOUT
    if [ "$slow" = yes ] && [ "$BATCH" -eq 4 ]; then batch=2; fi
    if [ "$slow" = yes ] && [ "$TIMEOUT" -eq 600 ]; then tmo=1800; fi

    if [ ! -x "$exe" ]; then
        echo "[SKIP] $name (binary not found: $exe)"
        SKIP=$((SKIP+1)); continue
    fi
    if [ ! -f "$ROOT/$cfg" ]; then
        echo "[SKIP] $name (config not found: $cfg)"
        SKIP=$((SKIP+1)); continue
    fi

    loops_args=""
    if [ -n "$LOOPS" ]; then loops_args="--loops $LOOPS"; fi
    log="$(mktemp /tmp/bench_batch_${name}.XXXXXX.log)"
    echo "[RUN ] $name batch=$batch config=$cfg"
    if (cd "$BIN_DIR" && timeout "$tmo" "./$name.out" "../$cfg" --batch "$batch" $loops_args) >"$log" 2>&1; then
        line="$(grep -o 'img/s=[0-9.]*' "$log" | tail -1)"
        echo "[PASS] $name $line"
        PASS=$((PASS+1))
    else
        rc=$?
        reason="exit=$rc"
        if [ "$rc" -eq 124 ]; then reason="timeout after ${tmo}s"; fi
        echo "[FAIL] $name ($reason), last output:"
        tail -n 6 "$log" | sed 's/^/    /'
        FAIL=$((FAIL+1)); FAILED_NAMES="$FAILED_NAMES $name"
    fi
    rm -f "$log"
done

echo ""
echo "== batch acceptance summary: PASS=$PASS FAIL=$FAIL SKIP=$SKIP (bin=$BIN_DIR, batch=$BATCH) =="
echo "   standalone (not batch-testable): $STANDALONE"
if [ "$FAIL" -ne 0 ]; then
    echo "   failed:$FAILED_NAMES"
    exit 1
fi
exit 0
