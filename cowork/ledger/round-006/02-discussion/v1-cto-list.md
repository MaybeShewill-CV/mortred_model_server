# 列表修订

- 轮次: 6
- 版本: v1
- 针对: 第 5 轮空列表之后，用户把上限改为 10

## P0
无

## P1
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P1-1 | RealESRGAN 输出 shape/体积不符时与空 tensor 共用 `MODEL_EMPTY_OUTPUT`，排障分不清 | 客户换导出权重或后端改 layout 后增强接口失败，运维只看 HTTP/状态码 | `validated_f32_first_output` 对契约失败会回 `MODEL_OUTPUT_CONTRACT_FAILED`；随后手工 shape/`element_count` 检查仍回 `MODEL_EMPTY_OUTPUT`，与空输出同码 | 错误 shape/体积与空输出返回不同状态码（或契约失败码），日志能直接区分两类故障 | `src/models/enhancement/real_esrgan.inl`（约 74–96 行）；`src/models/backend/f32_output.h`；`src/common/status_code.h` |
| P1-2 | drain 超时后仍析构成员，已借出 worker 租约未归还 | 模型推理挂死时停服/滚动重启（外层仍有 SIGKILL） | 析构对 async/worker drain 共用 `kDrainTimeoutMs`（3000ms）；超时只打 `drain timeout` 并 `return`，随后继续销毁 `_m_workers` 等成员，租约仍可能在 go 任务手里 | 超时后停服路径对未归还租约有可观察终态（不再在租约仍外借时直接拆成员），或明确移交外层杀进程前不再碰池内对象 | `src/server/base_server_impl.h`（约 68–99 行）；`src/server/worker_pool.h`（`kDrainTimeoutMs`、`drain` 超时 `return`） |
| P1-3 | 抠图/分割/分类三家空输出已 fail-closed，但无权重空输出契约未进专用 ctest（与已有深度空输出用例同类缺口） | 新上 modnet / bisenetv2 / resnet（及同结构分类）或改 postprocess 时靠手工打接口确认空输出行为 | 三家 `outputs.empty()` 均回 `MODEL_EMPTY_OUTPUT`；`test/CMakeLists.txt` 有 `mono_depth_output_contract_unittest` 等，无这三家空输出专用目标；`model_output_contract_unittest.cc` 测正常解码/错误 shape，未覆盖 `postprocess({})` | ctest 中有这三家（或合并一份）无权重空输出用例，失败时断言非 OK 且状态码/日志可区分 | `src/models/matting/modnet_matting.inl`；`src/models/scene_segmentation/bisenetv2.inl`；`src/models/classification/resnet.inl`；`test/CMakeLists.txt`；`test/model_output_contract_unittest.cc` |

## 附录
无

## 未采纳
| id | 来源表决 | 原因 |
| --- | --- | --- |
| cand-log-family | 候选 2（resnet/densenet/mobilenetv2 日志写「classification model」） | 已读到三处相同文案；一 catalog 一进程下进程/配置已点名家族，属日志措辞，不构成每机必做的易错手工步骤，本轮不升 P1 |
