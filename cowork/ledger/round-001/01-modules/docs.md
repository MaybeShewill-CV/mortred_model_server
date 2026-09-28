# 模块评分

- id: docs
- 权重: 10
- 分数: 7.8
- 读过的路径:
  - `docs/api-contract.md`
  - `docs/unsupported-boundaries.md`
  - `docs/deployment.md`

## 优势
- `docs/deployment.md` 把架构（gateway `:8080` / supervisor `:8787` / 模型 loopback）、cpu|gpu 决策、bootstrap/compose/tarball 三轨、`mortredctl next` 首小时车道、权重与 pack prepare/calibrate、鉴权 fail-closed、排障表和 §16 验收门写全，小团队能按文档把服务拉起来并做 live 验收。
- `docs/unsupported-boundaries.md` 把 Linux-only、CUDA 12.x + TensorRT 10.x、engine 本机本卡重建、RTDETR 脚手架、HF 作 ONNX 交换库等边界钉死，能挡住客户交付里最贵的假预期（错栈、跨机拷 engine、当 RTDETR 已上线）。
- `docs/api-contract.md` 讲清网关路由与鉴权顺序、统一 envelope、`status`↔HTTP 对照、队列 `429`、async `/jobs` 与已删除 `img_data`→422，联调和按状态码定位问题时有明确出处。

## 劣势
- `docs/api-contract.md` 人读摘要里 `results[].data` 只有分类示例，并把各任务 payload 推到 `docs/openapi.json` / `response_serializers.h`；检测/分割/抠图/OCR 等客户 SDK 不能只靠这一页定字段。
- 路径口径不一致：`docs/deployment.md` §1 图示客户端走 `/mortred_ai_server_v1/...`，`docs/api-contract.md` 主推 `/v1/models/{id}/…`（legacy `{server_uri}` 另述），现场写客户端容易接错入口。
- GPU 交付链在 `docs/deployment.md` §10 虽有 prepare、calibrate、`gpu_mem_mib` occupancy、doctor `--strict`，但步骤与「缺 stamp 则 spawn failed」分散；一两台卡同时伺候多项目时，仍易卡在「权重/引擎看起来有了但起不来」。
- `docs/deployment.md` FAQ 末尾（约离线权重说明处）出现无对应 `<summary>` 的游离 `</details>`，结构破损，降低按 FAQ 排障时的可信度。

## 依据
- 分数 7.8：三份必读齐备且上机/边界/契约主干可用（对应上述优势出处），但未到 9.0——多任务响应形态仍依赖 OpenAPI、路径表述分裂、GPU pack 操作性密度高、FAQ 结构瑕疵（对应劣势出处）。按口径：低于 9.0 必须有劣势；无未读必读路径，分数可高于 8.0。

## 未读
无
