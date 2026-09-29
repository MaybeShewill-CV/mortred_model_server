# 技术方案

- 问题: P1-3
- 版本: v1
- 对应列表: `cowork/ledger/round-002/02-p0-p1-final.md` P1-3

## 边界

只加厚 `mortredctl prepare` 的入口层：`scripts/mortredctl_prepare.sh` 不再纯 `exec`，在调用下游前做参数与 pack 文件门禁；失败时入口非 0，并在 stderr 点名缺哪一步或哪份产物。

不改 TensorRT 转换语义；不改 CUDA 12 / TensorRT 10；engine 仍本机 prepare。不做 P1-4。不启服务。不要求完整 GPU prepare。测试在 WSL 用 bash。一 catalog 一进程、gateway `:8080` 不变。

## 做法

### 推荐

`scripts/mortredctl_prepare.sh` 解析与下游一致的参数：`--pack`、`--force`、`--skip-ready`、`--require-ready`、`-h`/`--help`。默认 pack 与 `prepare_pack.sh` 相同（`MORTRED_PACK` 或 `conf/packs/demo.toml`）。`--help` 退出码 0。未知参数退出码非 0，stderr 含 `step:`。调用 `prepare_pack.sh` 之前检查 pack 是普通文件；缺失则退出码 1，stderr 含字面 `step: pack-file` 和该路径。通过后把原参数交给 `prepare_pack.sh` 并透传退出码。不改 `prepare_pack.sh` 的转换成功路径。入口不得再是整脚本 `exec prepare_pack.sh "$@"`。

### 否决

只改下游文案、入口继续纯 exec。把门禁搬进 C++ CLI。

## 改动面

- `scripts/mortredctl_prepare.sh`

## 明确不改

- `scripts/prepare_pack.sh` 转换成功路径。推荐：该文件无 diff。
- `src/`、`conf/`、P1-4、完整 GPU prepare。

## 验收标准

1. **AC1** `bash scripts/mortredctl_prepare.sh --pack /tmp/mortred-p1-3-missing-pack.toml` 退出码 1；stderr 含字面 `step: pack-file` 与路径 `/tmp/mortred-p1-3-missing-pack.toml`。
2. **AC2** `bash scripts/mortredctl_prepare.sh --help` 退出码 0。
3. **AC3** `bash scripts/mortredctl_prepare.sh --not-a-real-flag` 退出码非 0；stderr 含字面 `step:`。
4. **AC4** `rg -n 'exec .*prepare_pack' scripts/mortredctl_prepare.sh` 退出码 1（无匹配）。`git diff -- scripts/prepare_pack.sh` 为空。

## 自测命令

| 命令 | 期望退出码 |
| --- | --- |
| `bash scripts/mortredctl_prepare.sh --pack /tmp/mortred-p1-3-missing-pack.toml` | 1；stderr 含 `step: pack-file` 与该路径 |
| `bash scripts/mortredctl_prepare.sh --help` | 0 |
| `bash scripts/mortredctl_prepare.sh --not-a-real-flag` | 非 0；stderr 含 `step:` |
| `rg -n 'exec .*prepare_pack' scripts/mortredctl_prepare.sh` | 1 |

## RD 不得自行决定

不得改 TensorRT 成功路径、CUDA 12 / TensorRT 10。不得做完整 GPU prepare 或 P1-4。不得保留整脚本纯 `exec prepare_pack.sh "$@"`。缺 pack 的 stderr 必须含 `step: pack-file` 与路径，不得只靠下游 `[ERROR] pack not found`。未知参数必须由入口拒绝。
