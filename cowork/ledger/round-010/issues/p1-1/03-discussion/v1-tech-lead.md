# 技术方案

- 问题: P1-1
- 版本: v1
- 对应列表: cowork/ledger/round-010/02-p0-p1-final.md v2

## 边界

- 一个 catalog id 一个 OS 进程。不推翻 HTTP 契约和现有后端。
- 现有结构做不到完全不再写 C++。本条交付可复制模板，加上黑盒可判定「骨架已接入且未伪上线」。不是只改 TOML 的插件运行时，也不是全自动注册上线。
- 不支持边界不得做成功能：HTTP OCR 不做识别；RTDETR 保持脚手架、不进 factory / HTTP catalog；不做 MOT HTTP。
- 不扩大到 P1-2 或其他队列条目。不改 P0-1 已合入的 SAM 错码。

## 做法

### 推荐

以 `scripts/new_model.py` 和 `templates/model/` 为新架构的官方起点。钩子默认 `MODEL_NOT_IMPLEMENTED`。以已入库的 `rtdetr_detector` 为金丝雀：契约单测退出 0，且 `mortred-model-server.out --list` 不包含 RTDETR。文档写清仍须手填的清单和黑盒命令。

代价：仍要写钩子和手工注册。复制路径固定，「已接入骨架 / 未伪上线」可黑盒判定。

### 否决

1. 零 C++、只改 TOML 的通用插件运行时。会推翻现有头和 `.inl` 接法。
2. 脚手架自动注册进 HTTP catalog。半成品会被拉起，也和 RTDETR 未注册的边界冲突。

## 改动面

- `scripts/new_model.py` 的 `--check`、`--dry-run`、`--list-tasks` 可作门禁。不默认写 shared catalog。
- `templates/model/` 与 `tasks.json` 与现有一致性检查对齐。
- 金丝雀 `rtdetr_detector` 契约测保持可构建，不注册进 factory。
- 开发文档写清复制命令、仍须手填项、黑盒判定命令。

## 明确不改

- 进程模型、已上线模型的 HTTP 契约、后端集合、产品 toml 的 type 约定。
- RTDETR 不进 HTTP catalog。不做 OCR 识别、MOT HTTP。
- P0-1、P1-2 以及其他队列条目。

## 契约

未填钩子时返回 `MODEL_NOT_IMPLEMENTED`，不得用 OK 冒充已接入。金丝雀 RTDETR 必须继续不出现在 `--list` 里。

## 性能与资源

契约金丝雀和 `new_model.py --check` 不加载客户权重，不占长期 GPU session，不新增常驻进程。

## 验收标准

1. `python3 scripts/new_model.py --list-tasks` 退出码 0，且输出中能看到至少一条带 HTTP 的 CV task。
2. `python3 scripts/new_model.py --check` 退出码 0。
3. `python3 scripts/new_model.py --task object_detection --name probe_p1_1 --class ProbeP11Detector --backend tensorrt --dry-run` 退出码 0，且不落地写文件。
4. `rtdetr_detector_output_contract_unittest` 退出码 0。
5. `mortred-model-server.out --list` 退出码 0，输出中不包含 RTDETR 或 rtdetr 已服务项。
6. `python3 scripts/check_consistency.py` 退出码 0。

不把「零处手写 C++」「自动写入 catalog」「RTDETR 可 HTTP 推理」写成验收。

## 测试义务

QA 按验收 1–6 黑盒执行。白盒只核对脚手架没有写进 factory，且没有新增 OCR 识别或 MOT HTTP。

## 自测命令

```bash
python3 scripts/new_model.py --list-tasks
python3 scripts/new_model.py --check
python3 scripts/new_model.py --task object_detection --name probe_p1_1 --class ProbeP11Detector --backend tensorrt --dry-run
python3 scripts/check_consistency.py
```

期望退出码都是 0。`--dry-run` 不落地写文件。

## RD 不得自行决定

- 不得把 RTDETR 或脚手架注册进 factory、HTTP catalog 或默认 pack。
- 不得做成只改 TOML、零 C++ 的插件运行时。
- 不得实现 OCR 识别或 MOT HTTP。
- 不得用 OK 冒充未实现钩子。
- 不得顺带做 P1-2 或改 P0-1。
