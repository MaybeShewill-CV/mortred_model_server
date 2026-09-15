# SME 开箱可用清单（活看板）

目标：中小企业开箱即可用。只认稳定 ID（`SME-xx`），按序推进。  
对话口令：`清单` = 看本文件；`下一个` = 做第一个 `TODO`；`做 SME-xx` = 只做该项。  
流程：Mac 改 → push fix → WSL 验证 → 合 `main` → 删 fix → 勾本表。

状态：`TODO` | `DOING` | `DONE` | `BLOCKED`

| ID | 状态 | 事项 | 验收 |
|---|---|---|---|
| SME-14 | main @ b0c96f5 | _(merged; evidence pruned)_ |
| SME-16 | main @ 39a6b68 | _(merged; evidence pruned)_ |
| SME-13 | main @ 4a956a9 | _(merged; evidence pruned)_ |
| SME-12 | main @ 8798a0c | _(merged; evidence pruned)_ |
| SME-11 | main @ e66d8ed | _(merged; evidence pruned)_ |
| SME-10 | main @ e16fae2 | _(merged; evidence pruned)_ |
| SME-09 | main @ c00a233 | _(merged; evidence pruned)_ |
| SME-08 | main @ ec3fef7 | _(merged; evidence pruned)_ |
| SME-07 | main @ 4cfe6e5 | _(merged; evidence pruned)_ |
| SME-06 | main @ 6e1b7d1 | _(merged; evidence pruned)_ |
| SME-05 | main @ 1ebf4b4 | _(merged; evidence pruned)_ |
| SME-04 | main @ 7878a28 | _(merged; evidence pruned)_ release_dry_run.sh |
| SME-01 | DONE | WSL 归档全量 `tests-only`（+ 实际售卖 profile 的关键 GPU smoke） | 有可复现命令、退出码、失败清单 |
| SME-02 | DONE | 官方最短成功路径跑通并写入主航道文档：`install_deps` → 三 token →（GPU）`convert_trt` → `doctor --strict` → 一次推理 | 新人按文档能跟到绿 |
| SME-03 | DONE | `check_consistency`（或独立 checker）锁住 `ci.yml` dry-run/mock 契约 | 故意改错 CI 断言会被门禁抓住 |
| SME-04 | DONE | 发版 dry-run：GHCR 小写、tarball+`.sha256`、bootstrap mismatch 拒装 | 预发或本地演练通过 |
| SME-05 | DONE | `pack_file`：接线 `apply_pack` **或** 删掉/改诚实声称 | 配置与行为一致 |
| SME-06 | DONE | supervisor start/stop/restart 失败非恒 HTTP 200（或文档+客户端统一只认 body） | 监控/脚本可依赖约定 |
| SME-07 | DONE | 限流 vs 鉴权顺序：定产品意图并改代码或文档 | 行为与文档一致 |
| SME-08 | DONE | 修 `write_slot` 并发写 / `submit`–`stop` TOCTOU | 有单测或明确竞态门闩 |
| SME-09 | DONE | 批路径尊重剩余 deadline（与 HTTP timer 对齐） | 504 后不长时间占 worker |
| SME-10 | DONE | 干净机文档入口与 CMake 硬前置继续对齐（workflow/crypto/ORT 头等） | 干净树 configure 不踩已知坑 |
| SME-11 | DONE | 开箱收成**一条**主航道（三 token / listen / pack 校准；失败时给出下一条命令） | 只跟一条路径即可 |
| SME-12 | DONE | 写清不支持边界（RTDETR、Linux-only、必须重建 engine、TRT 10 only / 非 8·9） | 无假期待 |
| SME-13 | DONE | 删 zombie 注释（如 `web_console` / `ServerManager`） | 源码无过时声称 |
| SME-14 | DONE | `golden_drift_check.py --check` 进 CI + 重置基线 | PR 改 golden 必过零漂移门禁；基线与树一致 |
| SME-16 | DONE | convert_trt 与产品 TRT 10.x 钉对齐（拒 major&lt;10；env 仅探测覆盖） | TRT_VERSION_MAJOR=8 dry-run FAIL；探测失败无静默兜底；v100300 可解析 |

## 当前焦点

- **Next:** _(no open SME TODO)_
- **Notes:** Round-2 基线综合 8.0 / 开箱 6.8；企业开箱仍否。CHANGELOG 仅参考，以源码为准。

## 完成记录

| ID | 合入 main | 备注 |
|---|---|---|
| SME-01 | main @ 224a460 | WSL tests-only check 53/53 @ d51b0287 |
| SME-02 | main (this commit) | cpu shortest path OK; oob-main-path.md §0c (cpu source) |
