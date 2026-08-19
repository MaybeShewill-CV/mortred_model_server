#!/usr/bin/env bash
# verify_deployment.sh - 部署改造验收脚本（在目标 Linux + GPU 机器上执行）。
#
# 依次执行并汇总：
#   1. 部署脚本语法（bash -n / py_compile）
#   2. 清单 JSON / docker-compose YAML 合法性
#   3. convert_trt_engines.sh --list（引擎清单完整性）
#   4. fetch_weights.py --dry-run（权重清单可解析、HF 路径映射可用）
#   5. install_deps.sh --check（3rd_party 完整性；仅 --full 模式强制通过）
#   6. fetch_weights.py --check（本地权重 sha256；缺失权重时仅 --full 模式失败）
#
# 用法:
#   ./scripts/verify_deployment.sh            # --full：所有检查必须通过（目标机器）
#   ./scripts/verify_deployment.sh --basic    # 仅静态检查（未装依赖/无 GPU 的开发机）
#   ./scripts/verify_deployment.sh --verbose  # 打印每条检查的详细输出

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="full"
VERBOSE=0
FAILED=0

for arg in "$@"; do
    case "$arg" in
        --basic) MODE="basic" ;;
        --full) MODE="full" ;;
        --verbose) VERBOSE=1 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "[ERROR] unknown argument: $arg" >&2; exit 1 ;;
    esac
done

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
PY="$(resolve_python)" || { echo "[FAIL] missing a working python3/python"; exit 1; }

check() { # check <name> <cmd...>
    local name="$1"; shift
    if [ "$VERBOSE" -eq 1 ]; then
        echo "==> $name"
        "$@"
    else
        if "$@" >/dev/null 2>&1; then
            echo "  [ok]   $name"
        else
            echo "  [FAIL] $name"
            FAILED=$((FAILED+1))
        fi
    fi
}

echo "== Mortred 部署验收（mode=$MODE）=="

# 1) 脚本语法
for f in scripts/install_deps.sh scripts/convert_trt_engines.sh \
         scripts/docker_entrypoint.sh scripts/check_repo_clean.sh \
         scripts/clean_artifacts.sh scripts/setup_full_deps.sh; do
    check "bash -n $f" bash -n "$ROOT/$f"
done
check "py_compile fetch/gen/check" "$PY" -m py_compile \
    "$ROOT/scripts/fetch_weights.py" "$ROOT/scripts/gen_weights_manifest.py" \
    "$ROOT/scripts/check_consistency.py" "$ROOT/scripts/gen_openapi.py" "$ROOT/scripts/repo_toml.py"

# 2) 清单合法性
check "JSON: trt_engines/profiles/weights" "$PY" - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for f in ["conf/trt_engines.json",
          "conf/trt_profiles/lightglue_extractor.json",
          "conf/trt_profiles/lightglue_matcher.json",
          "conf/weights_manifest.json"]:
    json.loads((root / f).read_text(encoding="utf-8-sig"))
PY
if "$PY" -c "import yaml" >/dev/null 2>&1; then
    # 路径经 argv 传递（MSYS 会对独立参数做 Windows 路径转换；-c 字符串内不做）
    check "YAML: docker-compose.yml" "$PY" -c "import yaml,sys; yaml.safe_load(open(sys.argv[1], encoding='utf-8'))" "$ROOT/docker-compose.yml"
else
    echo "  [warn] yaml 模块缺失，跳过 docker-compose.yml 校验（pip install pyyaml）"
fi

# 3) 引擎清单
check "convert_trt_engines.sh --list" bash "$ROOT/scripts/convert_trt_engines.sh" --list

# 4) 权重清单 dry-run
check "fetch_weights.py --dry-run" "$PY" "$ROOT/scripts/fetch_weights.py" --dry-run

# 5) 3rd_party 完整性（--basic 模式下失败仅警告）
if [ "$MODE" = "full" ]; then
    check "install_deps.sh --check" bash "$ROOT/scripts/install_deps.sh" --check
else
    if bash "$ROOT/scripts/install_deps.sh" --check >/dev/null 2>&1; then
        echo "  [ok]   install_deps.sh --check"
    else
        echo "  [warn] install_deps.sh --check 未通过（basic 模式：依赖未安装属预期）"
    fi
fi

# 6) 本地权重校验（--basic 模式：抽样小文件验证机制，不做全量哈希）
if [ "$MODE" = "full" ]; then
    check "fetch_weights.py --check" "$PY" "$ROOT/scripts/fetch_weights.py" --check
else
    if "$PY" "$ROOT/scripts/fetch_weights.py" --check --only bpe_simple_vocab >/dev/null 2>&1; then
        echo "  [ok]   fetch_weights.py --check（抽样 bpe_simple_vocab）"
    else
        echo "  [warn] fetch_weights.py 抽样校验未通过（先运行 fetch_weights.py 下载权重）"
    fi
fi

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo "== 验收通过（mode=$MODE）=="
    exit 0
fi
echo "== 验收未通过：$FAILED 项失败（见上）=="
exit 1
