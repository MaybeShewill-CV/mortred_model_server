# 第2步表决

- 轮次: 1
- 针对版本: v1
- 结论: 待修订

## 逐条
| id | 表决 | 理由 |
| --- | --- | --- |
| P0-1 | 保留 | 使用者是一两台 GPU 机上交 TRT 包的小团队；场景是 `init→prepare→doctor→起 catalog`。已核对：产品 toml/`product_static_bs1`/`trt_engines.set_a` 多为 `*.static_bs1.fp16`，`onnx_sources` 的 yolov8s 仍写 `fp16in`，遗留 `trt_engines.json` 另套命名；`mortredctl doctor` 已能查 pack engine/occupancy，但缺件默认 WARN，仅 `--strict` 失败（复用检查、扩展默认 fail-closed）。验收可黑盒：默认命令非 0，且指出缺哪份 engine/stamp，不再「绿验收却起不来」。 |
| P0-2 | 保留 | 使用者是首次上机/换 token 起 supervisor+gateway+多模型的小团队。`api-contract` 已要求网关必须有独立 scrape Bearer（复用契约）；配空永久拒启、autostart 失败难定位、网关挂了仍起 model，会导致整棵入口无路由。验收可黑盒：配错 token 或网关未就绪时，supervisor 状态/API/`mortredctl doctor --live` 能看到永久失败原因与「勿继续起 model」终态，不必只翻 stderr。 |
| P1-1 | 拆开 | 一条里混了三类不可同验的事：(1)「只改 TOML 上新导出图/常见变体」无具体客户单、用平台化语言代替可观察行为，黑盒无法判定「常见」；(2) MOT：README 与 `unsupported-boundaries` 已写无 MOT HTTP，本条却要当交付能力，无哪一次交付必须突破的证据；(3) OCR：Zoo 仅 `dbnet`（检测），「整包识别」与「边界写清只检测」验收完全不同。应拆开后再各自写使用者与可判定验收；MOT 类勿编进本仓库新需求。 |
| P1-2 | 保留 | 使用者是一两块卡同时伺候多个 catalog 的交付方。已有 `calibrate`、`pack_occupancy`、doctor 门禁（复用）；`conf/packs` 仅 demo/yolov8/metric3d 且多只有 `worker_nums=1`、无写出的 `gpu_mem_mib`（扩展样本与写回链）。验收可黑盒：代表性 pack 带可复用 occupancy，换机 `calibrate→写回→doctor/spawn` 一条链可复跑成功。 |
| P1-3 | 保留 | 使用者是不用 compose、走本机 systemd/edge 的客户机运维。`deploy/*` 仍以注释列 useradd/`cmake --install`/enable（扩展为单一可复述入口）。验收可黑盒：一条 `mortredctl`/脚本后单元已 enable、服务可按文档起来，换机可复述成功。 |
| P1-4 | 保留 | 使用者是按 docs 写客户端对接 gateway 的业务侧。`api-contract` 主推 `/v1/models/{id}/…`（复用合同），`deployment.md` 仍示意 `/mortred_ai_server_v1/...`，任务 `results[].data` 示例不足。验收可黑盒：必读 docs 内单一推荐路径与任务 payload 示例可核对，联调不再猜入口。 |

## 不同意的条目
无
