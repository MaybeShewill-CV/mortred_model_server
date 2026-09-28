# 第2步签字

- 轮次: 1
- 针对版本: v2
- 结论: 同意

## 逐条
| id | 表决 | 理由 |
| --- | --- | --- |
| P0-1 | 保留 | 使用者是一两台 GPU 机交 TRT 包的小团队；场景是 `init→prepare→doctor→起 catalog`。改完后可黑盒验：默认命令非 0 并指出缺哪份 engine/stamp，不再「验收绿却起不来」。 |
| P0-2 | 保留 | 使用者是首次上机/换 token 起 supervisor+gateway 的小团队。改完后可黑盒验：配错 token 或网关未就绪时，supervisor 状态/API/`doctor --live` 能看到永久失败原因与「勿继续起 model」终态。 |
| P1-1 | 保留 | 使用者是按 Zoo/docs 接 OCR HTTP、交付「文字识别」类单的业务侧。仅保留文档/边界口径：OCR 仅检测、无识别；可黑盒核对必读 docs 与 Zoo 一致。MOT、「只改 TOML 上任意新模型」已进未采纳，符合边界。 |
| P1-2 | 保留 | 使用者是一两块卡伺候多 catalog 的交付方。改完后可黑盒验：代表性 pack 带可复用 occupancy，换机 `calibrate→写回→doctor/spawn` 一条链可复跑成功。 |
| P1-3 | 保留 | 使用者是不用 compose、走本机 systemd/edge 的客户机运维。改完后可黑盒验：一条入口后单元已 enable、服务可按文档起来，换机可复述成功。 |
| P1-4 | 保留 | 使用者是按 docs 写 gateway 客户端的业务侧。改完后可黑盒验：必读 docs 内单一推荐路径与任务 payload 示例可核对，联调不再猜入口。 |

## 不同意的条目
无
