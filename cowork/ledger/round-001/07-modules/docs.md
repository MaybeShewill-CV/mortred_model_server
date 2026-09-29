# 模块评分

- id: docs
- 权重: 10
- 分数: 8.5
- 读过的路径:
  - `docs/api-contract.md`
  - `docs/unsupported-boundaries.md`
  - `docs/deployment.md`

## 优势
- `docs/api-contract.md` 已给出分类对象、检测顶层数组、分割对象、OCR 检测顶层数组。
- 推荐入口已对齐：deployment 与 api-contract 主推 `POST /v1/models/{id}/infer`，遗留路径标明 legacy。
- `docs/unsupported-boundaries.md` 将 HTTP OCR 钉死为仅 `dbnet` 文字检测、无识别。
- `docs/deployment.md` §10 把 GPU pack 写成 copy → prepare → calibrate → doctor → start。

## 劣势
- `docs/deployment.md` §15 FAQ 末尾仍有游离 `</details>`。
- 人读契约仍未覆盖抠图 / 增强 / 深度等交付族的完整 `results[].data` 样例。
- GPU 多模型同机时，prepare / calibrate / occupancy / doctor 步骤仍集中在 §10 长文。

## 依据
- 分数 8.5：相对上轮 7.8，契约主线任务样例与推荐路径分裂已消。未到 9.0，因 FAQ 结构瑕疵、非主线任务人读样例缺口、GPU pack 操作性密度仍在。

## 未读
无
