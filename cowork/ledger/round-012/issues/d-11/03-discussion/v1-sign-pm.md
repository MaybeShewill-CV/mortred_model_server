# 第3步签字

- 问题: D-11
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 按部署文档找 `resnet`、并核对 `msocrnet` 是否被写成 Bench-only 的人，在仓库根目录跑验收 1、验收 2。两条都退出码 0，标准错误为空。验收 1 的标准输出是第 108 行与第 253 行全文，两处 cpu 名称都是 `resnet`。验收 2 的标准输出是 `CLIP, LightGlue, FastSAM, SAM prompt`，其中没有 `msocrnet`。这就是冻结句要看见的变化。catalog id、cpu 产品配置和 HTTP 路径已经是 `resnet` 与 `msocrnet`，本条只改这两行，不突破边界，不重开已关闭原句。
