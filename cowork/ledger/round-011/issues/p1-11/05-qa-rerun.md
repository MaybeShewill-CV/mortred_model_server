# 测试报告

- 结论: pass
- 方案版本: v2（2026-10-10 技术负责人条件）
- 分支: cowork/r11-p1-11-mask-overlap
- 工作树: G:\Codex\mortred-cowork-r11-p1-11
- commit: 无
- 环境: WSL Python 3.10.12，工作目录 `/mnt/g/Codex/mortred-cowork-r11-p1-11`。验收 3 的 `LD_LIBRARY_PATH` 含该工作树 `_lib` 与 `3rd_party/libs`，并去掉了 `MORTRED_UPDATE_GOLDEN` 与 `MORTRED_CI_REQUIRE_WEIGHTS`。被测代码 `2dceffc1b3e45a3c99d4f0e379ab1a6146433c19`。`core.autocrlf=true`。
- 缺的环境: 无
- 恢复条件: 无
- 未跑的验收: 1、2、4、5、6、7、9

## 用例

| 验收标准 | 类型 | 场景 | 命令 | 退出码 | 预期 | 实际 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 黑盒 | 四条数值通过 | 未设置 `MORTRED_UPDATE_GOLDEN`、`MORTRED_CI_REQUIRE_WEIGHTS`。`_bin/model_golden_test --gtest_filter=model_golden.bisenetv2_segmentation:model_golden.pphuman_segmentation:model_golden.modnet_matting:model_golden.ppmatting_matting` | 0 | 0，标准输出含 `NUMERICAL_PASSED` | `[  PASSED  ] 4 tests.`，标准输出一行 `NUMERICAL_PASSED`。耗时约 1213 / 227 / 2587 / 3672 ms | pass |
| 8 | 黑盒 | 黄金漂移检查 | `python3 scripts/golden_drift_check.py --check` | 0 | 0 | `zero drift: 30 cases and 26 golden files unchanged`。跑前 `test/golden/mobilenetv2_classification.json` 为 1004 个 LF、0 个 CR，未铺 HEAD blob，未执行 `git checkout`。跑后仍是 1004 个 LF、0 个 CR。没有铺回前、铺回后的另一次退出码 | pass |
| 8 | 白盒 | 键、名单、其余哈希、文件集 | 对照 `806813aa` 的 `test/golden_baseline.json` 与当前 `test/golden/` | 0 | 四个 png 键换成 json；`case_count` 30；指定四个哈希不变；不删其他黄金 | `case_count` 30，`golden_count` 26，`case_names` 与 `806813aa` 相同。去掉四份 png 键，加上四份 json 键，其余哈希无变化。`enlightengan_enhancement.png`、`fastsam_segmentation.png`、`sam_prompt_prediction.png`、`yolov8_detection.json` 与 `806813aa` 相同。磁盘上 26 个文件与基线键一致，四份 png 不在，四份 json 在 | pass |

## 缺陷

无

## 不适用

- 错误 catalog id、未就绪模型：本条不改 HTTP catalog。
- 空输入、通道或色彩与预处理不符：验收没有这条。尺寸不符在验收 9，这次未跑。
- 检测框坐标和分类分数：本条不改检测和分类。
- HTTP 超时、参数缺失：方案写明无 HTTP 契约变化。
- TensorRT 同规则比较：方案定为 P1-12。本条不加载产品 toml，不生成 engine。
- `full` 构建、`msocrnet`、19 条同进程过滤、`golden_drift_check.py --record`：方案禁止，未执行。
- `MORTRED_UPDATE_GOLDEN=1`：方案写明那一次退出 0 不算验收 3。本次没有执行更新黄金。

## 残余风险

验收 1、2、4、5、6、7、9 这次没有重跑。上一份报告里的白盒测试义务（比较实现、类型不符文案、黄金类别和前景统计）这次没有重核。类型不符按方案只留白盒，这次没有改模型输出类型去触发。`core.autocrlf=true` 仍在；这次工作区该文件是 LF，`--check` 退出码 0。工作树交回时 `git status --porcelain` 为空，HEAD 仍是 `2dceffc1b3e45a3c99d4f0e379ab1a6146433c19`。
