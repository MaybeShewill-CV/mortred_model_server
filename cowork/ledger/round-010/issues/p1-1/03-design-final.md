# 技术方案

- 问题: P1-1
- 版本: v1
- 对应列表: cowork/ledger/round-010/02-p0-p1-final.md P1-1
- 作者: 编排方（单一执行方模式；通用子代理按角色卡确认，见 03 确认节）

## 边界

只处理"检测参数 `clip_boxes` 是无效且与实现相反的开关"这一件事。`min_box_area_px` 的语义与实现**不在本方案范围内**——已逐行复核确认它在 v5/v6/v7/v8 都于网络空间过滤（`yolo_decode.h:24,59,118,172`；v8 在 `yolov8_detector.inl:75-76` 过滤、`:85` 才 unmap），与文档和用例名一致。

现状（已核实）：
- `DetectionParams::clip_boxes`（`detection_params.h:26`）只在 `:100-108` 被解析赋值，`src/`、`test/`、`conf/` 内**零使用者**；
- letterbox 反解 `unmap_letterbox_bbox`（`request_geometry.h:127-144`）对每个框**恒定**夹到 `[0, source]`（`:139-142`），所以实际行为永远是"已裁剪"，该键既不能开启也不能关闭任何东西；
- 因此"`clip_boxes=false`（默认）表示不裁剪"这句话与实现相反。

## 做法

### 推荐

按本文件既有的"已移除/不受支持键"约定处理（参照 `detection_params.h:43-49` 对 `input_node_size`、`model_score_threshold` 的处置）：

1. 删除 `DetectionParams` 的 `clip_boxes` 字段（`detection_params.h:26`）与其解析分支（`:100-108`）。
2. 在 `parse()` 已有的"不受支持键"分支里加一条 `clip_boxes` 的显式拒绝，错误信息说明真实语义，例如：`params key 'clip_boxes' is not supported; detections are always clipped to the source image after letterbox unmap`。
3. 在 `test/detection_params_unittest.cc` 增加一条用例：`clip_boxes = true` 的配置使 `parse` 返回 false 且错误信息含 `not supported`；同时断言 `min_box_area_px` 仍被正常解析（防止顺手改坏邻居）。

这样做的收益：配置面不再包含无法生效的键；配错的运维会在**启动时**拿到解释性错误，而不是以为自己在控制裁剪。

### 否决

- 实现"可选裁剪"（用 `clip_boxes=false` 关闭反解处的恒定裁剪）：这会把"恒定裁剪"这一既有正确行为改成可关闭，属于行为变更且风险更大；也没有任何使用者要求它。
- 只删字段、保留静默忽略：`clip_boxes = true` 会被无声吞掉，运维无法发现自己配了无效键——与完成标准里"配置错误要能看见"的方向相反。
- 用一条注释"该键无效"了事：仍然留着字段与解析分支，`grep` 仍能命中，契约面依旧骗人。

## 改动面

- `src/models/object_detection/detection_params.h`（删字段 + 删解析分支 + 加显式拒绝）
- `test/detection_params_unittest.cc`（新增 1–2 条用例）

## 明确不改

- `min_box_area_px` 的字段、默认值、`passes_min_box_area`、各检测器的过滤位置与坐标空间。
- `unmap_letterbox_bbox` 的恒定裁剪行为。
- 检测任务的 `param_spec` 既有键（`score_threshold` / `nms_threshold` / `top_k`）。
- 任何 `conf/**`、`docs/openapi.json`、生成脚本。

## 契约

- 模型 TOML 的检测参数不再接受 `clip_boxes`；带该键的配置在模型初始化阶段失败并给出可操作错误。
- 其余检测参数行为不变；反解仍恒定裁剪到源图边界。

## 性能与资源

无（只删一个 bool 字段与一段解析分支）。

## 验收标准

- A1 `grep -rn "clip_boxes" src/ test/ conf/ docs/openapi.json` 只命中 `detection_params.h` 中"显式拒绝"的那几行（字符串字面量与错误信息），**不再有**字段声明或赋值给 `out->clip_boxes`；`conf/` 与 `docs/` 内 0 命中。
- A2 新增用例通过：`clip_boxes = true` → `DetectionParams::parse` 返回 false，`error` 含 `not supported`。
- A3 `./_bin/detection_params_unittest` 退出 0 全过；`./_bin/detector_common_unittest`、`./_bin/object_detection_output_contract_unittest` 退出 0 全过（证明没有波及检测语义）。
- A4 问题分支提交只含上述 2 个文件；`git show --stat` 不含 `cowork/ledger/`、不含构建产物与 `src/common/mortred_version.h`。

## 测试义务

- 每条验收标准一行，给出真实命令与退出码。
- 白盒：确认删除后 `DetectionParams` 内不再有名为 `clip_boxes` 的成员；确认 `min_box_area_px` 相关既有用例**未被修改**（`git diff` 中不出现）。
- 反例：构造 `clip_boxes = "yes"`（字符串）也应被拒绝且错误信息指向该键。

## 自测命令

```bash
cd /mnt/g/Codex/mortred_wt_r10b
cmake --build build --target detection_params_unittest detector_common_unittest object_detection_output_contract_unittest -j8   # 期望 0
./_bin/detection_params_unittest              ; echo $?   # 期望 0
./_bin/detector_common_unittest               ; echo $?   # 期望 0
./_bin/object_detection_output_contract_unittest ; echo $?   # 期望 0
grep -rn "clip_boxes" src/ test/ conf/ docs/openapi.json
```

（`object_detection_output_contract_unittest` 链接 `models`，首次构建较慢；三个二进制都以仓库根为 CWD 运行。）

## RD 不得自行决定

- 不得实现"可选裁剪"或改动 `unmap_letterbox_bbox` 的行为。
- 不得改 `min_box_area_px`、`passes_min_box_area` 或任何检测器的过滤位置。
- 不得删除或放宽既有检测用例。
- 不得把 `cowork/ledger/`、构建产物、`src/common/mortred_version.h` 带进提交。
