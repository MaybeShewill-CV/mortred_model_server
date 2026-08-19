#!/usr/bin/env bash
# convert_trt_engines.sh - 按 conf/trt_engines.json 生成硬件适配的 TensorRT engine。
#
# 背景：config 期望的 .engine 与用户 GPU 架构 / TRT 版本强相关，必须按本机生成
# （原权重包中的引擎可能与当前 TRT 版本错配）。本脚本把 onnx 源转换为 config
# 引用路径下的 engine，替换版本错配文件。
#
# 用法（在仓库根目录执行）:
#   ./scripts/convert_trt_engines.sh                 # 转换缺失的引擎
#   ./scripts/convert_trt_engines.sh --force         # 全部重新转换（覆盖已有）
#   ./scripts/convert_trt_engines.sh --list          # 打印清单（不转换）
#   ./scripts/convert_trt_engines.sh --only yolov8   # 只转换路径含 yolov8 的条目
#   ./scripts/convert_trt_engines.sh --converter _bin/onnx2trt_converter.out
#
# 依赖: full build 产物 _bin/onnx2trt_converter.out；缺失的 onnx 用
#       scripts/fetch_weights.py 下载。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/conf/trt_engines.json"
CONVERTER="${CONVERTER:-$ROOT/_bin/onnx2trt_converter.out}"
FORCE=0
ONLY=""
MODE="convert"

usage() {
    sed -n '2,16p' "$0"
    exit 0
}

fail() { echo "[ERROR] $*" >&2; exit 1; }

require_cmd() { command -v "$1" >/dev/null 2>&1 || fail "missing command: $1"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --list) MODE="list"; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        --converter) CONVERTER="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) fail "unknown argument: $1 (see --help)" ;;
    esac
done

[ -f "$MANIFEST" ] || fail "manifest not found: $MANIFEST"

# 解析一个真正可执行的 python（跳过 PATH 中的失效存根，如 WindowsApps 的 python3 别名）
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

# 用 python 解析清单（jq 不一定存在），输出 TSV: model<TAB>onnx<TAB>engine<TAB>fp<TAB>profile
# 写临时文件而非进程替换：Windows Git Bash 下 mapfile+heredoc+进程替换组合不可靠
TMPLIST="$(mktemp)" || fail "mktemp failed"
if ! "$PY" - "$MANIFEST" "$ONLY" >"$TMPLIST" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8-sig"))
only = sys.argv[2]
for e in manifest.get("engines", []):
    path = e.get("engine", "")
    if only and only.lower() not in path.lower():
        continue
    print("\t".join([e.get("model", ""), e.get("onnx", ""), path,
                     str(e.get("fp", 0)), e.get("profile") or ""]))
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

if [ "$MODE" = "list" ]; then
    printf "%-32s %-70s %s\n" "MODEL" "ONNX" "ENGINE"
    for line in "${ENTRIES[@]}"; do
        IFS=$'\t' read -r model onnx engine fp profile <<<"$line"
        printf "%-32s %-70s %s\n" "$model" "$onnx" "$engine"
    done
    echo "total: ${#ENTRIES[@]}"
    exit 0
fi

[ -x "$CONVERTER" ] || fail "converter not found/executable: $CONVERTER (先完成 full build)"

converted=0; skipped=0; missing_onnx=0
for line in "${ENTRIES[@]}"; do
    IFS=$'\t' read -r model onnx engine fp profile <<<"$line"
    onnx_path="$ROOT/$onnx"
    engine_path="$ROOT/$engine"
    if [ ! -f "$onnx_path" ]; then
        echo "[skip] $model: onnx 缺失 $onnx (先运行 ./scripts/fetch_weights.py --only $model)"
        missing_onnx=$((missing_onnx+1))
        continue
    fi
    if [ -f "$engine_path" ] && [ "$FORCE" -eq 0 ]; then
        echo "[skip] $model: engine 已存在 $engine (加 --force 重新转换)"
        skipped=$((skipped+1))
        continue
    fi
    mkdir -p "$(dirname "$engine_path")"
    args=("$onnx_path" "$engine_path" "$fp")
    if [ -n "$profile" ]; then
        [ -f "$ROOT/$profile" ] || fail "$model: profile 缺失 $profile"
        args+=("$ROOT/$profile")
        echo "[convert] $model: fp=$fp profile=$profile"
    else
        echo "[convert] $model: fp=$fp"
    fi
    if "$CONVERTER" "${args[@]}"; then
        converted=$((converted+1))
        echo "  -> $engine"
    else
        echo "[FAIL] $model: 转换失败（退出码 $?）"
    fi
done

echo ""
echo "== 完成: 转换 $converted, 跳过(已存在) $skipped, 缺 onnx $missing_onnx"
[ "$missing_onnx" -eq 0 ] || echo "== 提示: 缺失 onnx 请先运行 scripts/fetch_weights.py 下载"
exit 0
