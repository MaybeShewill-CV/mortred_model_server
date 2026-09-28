# 第3步签字

- 问题: P0-1
- 方案版本: v1
- 结论: 不同意
- 条件: 在 AC3、AC4、AC6 正文各补一句：判定前将当前 pack 设为 `conf/packs/yolov8.toml`（例如设置 `MORTRED_PACK` 指向该文件），再执行对应的无 `--strict` 的 `mortredctl doctor` 或 `mortredctl init --profile gpu`；不得在未绑定该 pack、仍走默认 demo 的口径下宣称上述三条已通过。
- 理由: 冻结可观察变化要求客户检测 TRT 包上「单一真源 + 未就绪默认非 0 并点名缺 engine/stamp」。AC1/AC2/AC5/AC7/AC8 能黑盒核对路径一致、权重失败、cpu 不链 TRT、就绪后 doctor/`pack_trt --check` 变绿；但 AC3/AC4/AC6 只写「当前 pack」而未点名 yolov8 pack，默认 demo 无 TRT 时现场无法对冻结场景做同样判定，故不能签字同意。
