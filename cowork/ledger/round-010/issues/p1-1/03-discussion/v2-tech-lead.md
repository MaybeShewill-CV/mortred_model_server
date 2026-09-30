# 技术方案

- 问题: P1-1
- 版本: v2
- 对应列表: cowork/ledger/round-010/02-p0-p1-final.md v2

## 边界

- 一个 catalog id 一个 OS 进程。不推翻 HTTP 契约和现有后端。
- 本条交付是可复制脚手架，加上黑盒可判定「骨架已接入且未伪上线」。不是只改 TOML 的插件运行时，也不是自动注册上线。
- HTTP OCR 不做识别。RTDETR 保持脚手架、不进 factory / HTTP catalog。不做 MOT HTTP。
- 不扩大到其他队列条目。不改已合入的 SAM 错码。
- v1 验收 1–6 在现有仓库已满足，不算本条增量。v2 必须落地：对 classification 实际生成脚手架，未填钩子返回 `MODEL_NOT_IMPLEMENTED`，不进 `--list`，不写进产品 catalog，测完清理。

## 做法

### 推荐

以 classification 为官方复制样例。验收探针实际生成脚手架（不得只用 `--dry-run`，不得 `--apply-catalog`），跑契约测看到 `MODEL_NOT_IMPLEMENTED`，`--list` 不含该探针，factory 无新增 Entry，测完删除生成物。开发文档写出可逐步执行的复制命令、仍须手填清单和黑盒判定。

### 否决

1. 继续用 v1 已绿的门禁当本条交付。
2. 零 C++、只改 TOML 的插件运行时。
3. 对探针使用 `--apply-catalog` 或写进 factory。
4. 用 OCR 或 MOT 当样例。
5. 只改文档、不实际生成。

## 改动面

- `scripts/new_model.py` 或同目录验收探针脚本：生成 → 挂契约测 → 跑测 → 断言未进 catalog/`--list` → 删除生成物。
- 开发文档中英两份写出 classification 复制命令和手填清单。
- 临时 CMake 挂接只服务验收，结束后产品树不得残留探针源文件和 catalog 行。

## 明确不改

进程模型、已上线模型契约、后端集合、产品 toml 的 type 约定、RTDETR 注册状态、P0-1 与其他队列条目。

## 契约

未填钩子必须返回 `MODEL_NOT_IMPLEMENTED`。探针与 RTDETR 都不得出现在 `--list`。HTTP 与产品 catalog 不新增可拉起的半成品。

## 性能与资源

验收不加载客户权重，不占长期 GPU session，不新增常驻进程，不新增产品 engine。

## 验收标准

1. 开发文档有 classification 的复制命令，不含 `--dry-run`，不含 `--apply-catalog`，并列出仍须手填清单和黑盒判定命令。
2. 按该命令实际生成脚手架，退出码 0，文件系统出现头、`.inl` 和契约测源。
3. 探针的 `*_output_contract_unittest` 退出码 0，且未填钩子路径为 `MODEL_NOT_IMPLEMENTED`。
4. 探针已生成且未写入 factory 时，`mortred-model-server.out --list` 退出码 0，输出不含该探针。
5. 验收期间及清理后，`src/factory/classification_task.h` 无该探针 Entry。不得用 `--apply-catalog` 完成验收。
6. 清理后工作区无探针残留；再次 `--list` 仍无该探针；`python3 scripts/check_consistency.py` 退出码 0。
7. 回归：`python3 scripts/new_model.py --check` 退出码 0；`rtdetr_detector_output_contract_unittest` 退出码 0；`--list` 仍不含 RTDETR。

## 测试义务

QA 必须实际执行复制命令，跑契约测，核对 `--list` 与 factory，再核对清理后无残留。白盒确认未调用 `--apply-catalog`，未新增 OCR 识别或 MOT HTTP。

## 自测命令

```bash
python3 scripts/new_model.py --task classification --name p1_1_cls_probe --class P11ClsProbe --backend mnn
```

期望退出码 0。随后构建并运行探针契约测，期望退出码 0，未填钩子为 `MODEL_NOT_IMPLEMENTED`。`--list` 期望退出码 0 且不含 `p1_1_cls_probe`。清理后 `python3 scripts/check_consistency.py` 期望退出码 0。

若有一键探针脚本，退出码 0 当且仅当验收 2–6 全部成立。文档仍须保留可逐步手跑的等价命令。

## RD 不得自行决定

- 不得用 v1 验收或仅 `--dry-run` 宣称完成。
- 不得 `--apply-catalog`，不得把探针留在产品 catalog。
- 不得改选 OCR/MOT，不得实现识别或 MOT HTTP。
- 不得把 RTDETR 注册为已上线检测器。
- 不得做成零 C++ 插件运行时。
- 不得用 OK 冒充未实现钩子。
- 不得顺带做其他队列条目。
- 探针身份若要偏离 `--task classification --name p1_1_cls_probe --class P11ClsProbe --backend mnn`，必须回到方案修订。
