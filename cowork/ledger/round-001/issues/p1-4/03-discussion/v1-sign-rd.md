# 第3步签字

- 问题: P1-4
- 方案版本: v1
- 结论: 不同意
- 条件: 自测命令必须按文件分别判定，不得用一次 `rg -q` 扫两份 `api-contract` 就算过。A2 须有可复现命令核对两份 `deployment` 的 `:8080` 用途列含字面量 `POST /v1/models/{id}/infer`，且该行不得把未标明遗留的 `/mortred_ai_server_v1` 当作推荐推理入口。A4 须能分别核对四段 JSON：分类对象含 `class_id`/`category`/`scores`；检测为含 `bbox`+`detail_infos` 的数组项（不得用 OCR 块里的同名字段冒充）；分割对象含 `image`+`colorized_mask`；OCR 检测块含 `polygon` 且该块不出现字段名 `"text"`。A1 正文删除「或等价」口径，或把自测改成与该口径一致，不得一边允许等价写法一边用 `-F 'POST /v1/models/{id}/infer'` 卡死。
- 理由: 推荐 `POST /v1/models/{id}/infer` 已是现网 gateway 路径，catalog 四例与 OpenAPI/`response_serializers.h` 的分类/检测/分割/OCR 检测字段能对上，文档四份收口可在一个 worktree 里改完且不必动 `src`/OpenAPI/启服务。不能同意是因为现列 rg 与 A1–A4 对不上：A2 无端口表门；A4 现命令在英文信封已有 `class_id`/`scores` 时，加一段带 `bbox` 的 OCR 加 `colorized_mask` 即可全绿，抽读又不在自测命令里，WSL 只用方案列出的 rg/test 关不掉正文验收。
