#!/usr/bin/env bash
# convert_trt_engines.sh - 使用外部 trtexec（TensorRT 官方 CLI）按 conf/trt_engines.json
# 生成硬件适配的 TensorRT engine。
#
# 背景：config 期望的 .engine 与用户 GPU 架构 / TRT 版本强相关，必须按本机生成
# （原权重包中的引擎可能与当前 TRT 版本错配）。本脚本用 trtexec 把 onnx 源转换为
# config 引用路径下的 engine，替换版本错配文件。自研转换器已移除。
#
# 用法（在仓库根目录执行）:
#   ./scripts/convert_trt_engines.sh                  # 转换缺失的引擎
#   ./scripts/convert_trt_engines.sh --force          # 全部重新转换（覆盖已有）
#   ./scripts/convert_trt_engines.sh --list           # 打印清单（不需要 trtexec）
#   ./scripts/convert_trt_engines.sh --only yolov8    # 只转换路径含 yolov8 的条目
#   ./scripts/convert_trt_engines.sh --strict         # 首个失败立即退出（CI 友好）
#   ./scripts/convert_trt_engines.sh --check-engines  # 只校验已有引擎（存在+非空）
#   ./scripts/convert_trt_engines.sh --dry-run        # 只打印将执行的命令行，不转换
#   ./scripts/convert_trt_engines.sh --trtexec /path/to/trtexec
#
# trtexec 查找顺序：$TRTEXEC（env/--trtexec）→ 3rd_party/bin/trtexec
#                  （install_deps.sh --nvidia 安装）→ PATH → /usr/src/tensorrt/bin/trtexec
# 依赖: 缺失的 onnx 用 scripts/fetch_weights.py 下载。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/conf/trt_engines.json"
LIB_DIR="$ROOT/3rd_party/libs"
TRTEXEC="${TRTEXEC:-}"
FORCE=0
ONLY=""
MODE="convert"
STRICT=0
# 与旧自研转换器 6GB workspace 对齐；可用 TRTEXEC_WORKSPACE 覆盖
WORKSPACE_STR="${TRTEXEC_WORKSPACE:-6G}"

usage() {
    sed -n '2,21p' "$0"
    exit 0
}

fail() { echo "[ERROR] $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --list) MODE="list"; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        --strict) STRICT=1; shift ;;
        --check-engines) MODE="check-engines"; shift ;;
        --dry-run) MODE="dry-run"; shift ;;
        --trtexec) TRTEXEC="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) fail "unknown argument: $1 (see --help)" ;;
    esac
done

[ -f "$MANIFEST" ] || fail "manifest not found: $MANIFEST"

# ---- 解析一个真正可执行的 python（跳过 PATH 中的失效存根，如 WindowsApps 别名） ----
resolve_python() {
    local cand p
    for cand in python3 python py; do
        if command -v "$cand" >/dev/null 2>&1; then
            p="$(command -v "$cand")"
            if "$p" -c 'import sys' >/dev/null 2>&1; then
                echo "$p"
                return 0
            fi
        fi
    done
    return 1
}
PY="$(resolve_python)" || fail "missing a working python3/python (needed to parse $MANIFEST)"

# ---- 用 python 解析清单 + profile，输出 TSV: model<TAB>onnx<TAB>engine<TAB>fp<TAB>shape_flags ----
# 写临时文件而非进程替换：Windows Git Bash 下 mapfile+heredoc+进程替换组合不可靠
TMPLIST="$(mktemp)" || fail "mktemp failed"
if ! "$PY" - "$ROOT" "$MANIFEST" "$ONLY" >"$TMPLIST" <<'PY'
import json, sys
from pathlib import Path
root, manifest_path, only = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))

def dims(v):
    return "x".join(str(d) for d in v)

for e in manifest.get("engines", []):
    path = e.get("engine", "")
    if only and only.lower() not in path.lower():
        continue
    flags = ""
    if e.get("profile"):
        prof = json.loads((root / e["profile"]).read_text(encoding="utf-8-sig"))
        mins, opts, maxs = [], [], []
        for b in prof:  # 多 binding 通用（lightglue matcher 4 个 binding）
            mins.append(f'{b["name"]}:{dims(b["min"])}')
            opts.append(f'{b["name"]}:{dims(b["opt"])}')
            maxs.append(f'{b["name"]}:{dims(b["max"])}')
        flags = ("--minShapes=" + ",".join(mins) + " "
                 "--optShapes=" + ",".join(opts) + " "
                 "--maxShapes=" + ",".join(maxs))
    print("\t".join([e.get("model", ""), e.get("onnx", ""), path,
                     str(e.get("fp", 0)), flags]))
PY
then
    rm -f "$TMPLIST"
    fail "解析清单失败: $MANIFEST"
fi
mapfile -t ENTRIES < "$TMPLIST"
rm -f "$TMPLIST"
if [ "${#ENTRIES[@]}" -eq 0 ]; then
    [ -n "$ONLY" ] && fail "没有匹配 '$ONLY' 的条目（见 --list）"
    fail "清单为空: $MANIFEST"
fi

# ---- list：只读清单，不需要 trtexec ----
if [ "$MODE" = "list" ]; then
    printf "%-32s %-70s %s\n" "MODEL" "ONNX" "ENGINE"
    for line in "${ENTRIES[@]}"; do
        IFS=$'\t' read -r model onnx engine fp flags <<<"$line"
        printf "%-32s %-70s %s\n" "$model" "$onnx" "$engine"
    done
    echo "total: ${#ENTRIES[@]}"
    exit 0
fi

# ---- check-engines：只校验已有引擎（存在+非空），不需要 trtexec ----
if [ "$MODE" = "check-engines" ]; then
    bad=0
    for line in "${ENTRIES[@]}"; do
        IFS=$'\t' read -r model onnx engine fp flags <<<"$line"
        if [ -s "$ROOT/$engine" ]; then
            echo "  [ok] $engine"
        else
            echo "  [!!] $model: 引擎缺失或为空 $engine"
            bad=$((bad+1))
        fi
    done
    [ "$bad" -eq 0 ] || exit 1
    exit 0
fi

# ---- 解析 trtexec（convert 模式必需；dry-run 缺 trtexec 时按 TRT 8 语法降级输出） ----
if [ -z "$TRTEXEC" ]; then
    for cand in "$ROOT/3rd_party/bin/trtexec" \
                "$(command -v trtexec 2>/dev/null || true)" \
                "/usr/src/tensorrt/bin/trtexec" \
                "${TENSORRT_ROOT:-/usr/src/tensorrt}/bin/trtexec"; do
        [ -n "$cand" ] && [ -x "$cand" ] && { TRTEXEC="$cand"; break; }
    done
fi
if [ -z "$TRTEXEC" ] && [ "$MODE" != "dry-run" ]; then
    fail "trtexec not found（sudo ./scripts/install_deps.sh --nvidia 可安装；或 --trtexec /path/to/trtexec）"
fi

# ---- TRT 主版本探测：8.x 用 --workspace=<字节>；9+/10 用 --memPoolSize=workspace:<大小> ----
TRT_MAJOR="${TRT_VERSION_MAJOR:-}"
if [ -z "$TRT_MAJOR" ] && [ -n "$TRTEXEC" ]; then
    TRT_MAJOR="$("$TRTEXEC" --help 2>&1 | grep -m1 -oE 'version:?[[:space:]]*[0-9]+' | grep -oE '[0-9]+$' || true)"
fi
if [ -z "$TRT_MAJOR" ]; then
    if [ "$MODE" = "dry-run" ]; then
        TRT_MAJOR=8
        echo "[warn] dry-run: 无法探测 TRT 版本，按 8.x 语法输出（TRT_VERSION_MAJOR 可覆盖）" >&2
    else
        fail "无法探测 TRT 版本（设置 TRT_VERSION_MAJOR 或改用正确的 trtexec）"
    fi
fi

# vendored trtexec 需要 3rd_party/libs 在动态库路径上
if [[ "$TRTEXEC" == "$ROOT/3rd_party/"* ]]; then
    export LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

size_to_bytes() {
    local s="$1" n u
    n="${s%[KkMmGg]}"
    u="${s: -1}"
    case "$u" in
        K|k) echo $((n*1024)) ;;
        M|m) echo $((n*1024*1024)) ;;
        G|g) echo $((n*1024*1024*1024)) ;;
        *) echo "$n" ;;
    esac
}

if [ "$TRT_MAJOR" -ge 9 ]; then
    WS_FLAG="--memPoolSize=workspace:$WORKSPACE_STR"
else
    WS_FLAG="--workspace=$(size_to_bytes "$WORKSPACE_STR")"
fi

converted=0; skipped=0; missing_onnx=0; failed=0
declare -a failed_models=()
for line in "${ENTRIES[@]}"; do
    IFS=$'\t' read -r model onnx engine fp flags <<<"$line"
    onnx_path="$ROOT/$onnx"
    engine_path="$ROOT/$engine"
    if [ ! -f "$onnx_path" ]; then
        echo "[skip] $model: onnx 缺失 $onnx（先运行 ./scripts/fetch_weights.py --only $model）"
        missing_onnx=$((missing_onnx+1))
        continue
    fi
    if [ -f "$engine_path" ] && [ "$FORCE" -eq 0 ]; then
        echo "[skip] $model: engine 已存在 $engine（加 --force 重新转换）"
        skipped=$((skipped+1))
        continue
    fi
    case "$fp" in
        0) fp_flag="" ;;
        1) fp_flag="--fp16" ;;
        *)
            echo "[FAIL] $model: 未知 fp=$fp（仅支持 0=FP32 / 1=FP16）"
            failed=$((failed+1)); failed_models+=("$model")
            [ "$STRICT" -eq 1 ] && exit 1
            continue ;;
    esac
    # flags 为 profile 翻译产物（空格分隔的 --minShapes/--optShapes/--maxShapes）
    # shellcheck disable=SC2206
    args=(--onnx="$onnx_path" --saveEngine="$engine_path" --buildOnly)
    [ -n "$fp_flag" ] && args+=("$fp_flag")
    args+=($flags)
    args+=("$WS_FLAG")
    echo "[convert] $model: fp=$fp${flags:+ profile=$flags}"
    if [ "$MODE" = "dry-run" ]; then
        echo "  cmd: $TRTEXEC ${args[*]}"
        continue
    fi
    mkdir -p "$(dirname "$engine_path")"
    if out="$("$TRTEXEC" "${args[@]}" 2>&1)"; then
        if [ -s "$engine_path" ]; then
            converted=$((converted+1))
            echo "  -> $engine"
        else
            failed=$((failed+1)); failed_models+=("$model")
            echo "[FAIL] $model: trtexec 返回 0 但引擎缺失或为空"
            [ "$STRICT" -eq 1 ] && exit 1
        fi
    else
        rc=$?
        failed=$((failed+1)); failed_models+=("$model")
        echo "[FAIL] $model: trtexec 失败（退出码 $rc）"
        echo "$out" | tail -n 15
        [ "$STRICT" -eq 1 ] && exit 1
    fi
done

echo ""
echo "== 完成: 转换 $converted, 跳过(已存在) $skipped, 缺 onnx $missing_onnx, 失败 $failed"
if [ "$failed" -gt 0 ]; then
    echo "== 失败条目:"
    for m in "${failed_models[@]}"; do
        echo "   - $m"
    done
    echo "== 提示: 加 --strict 使首个失败即停止；动态输入模型若失败，请在 conf/trt_profiles/ 补 profile 并写入 conf/trt_engines.json"
    exit 1
fi
[ "$missing_onnx" -eq 0 ] || echo "== 提示: 缺失 onnx 请先运行 scripts/fetch_weights.py 下载"
exit 0
