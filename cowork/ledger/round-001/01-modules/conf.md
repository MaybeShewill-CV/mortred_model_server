# 模块评分

- id: conf
- 权重: 10
- 分数: 6.8
- 读过的路径: `cowork/ledger/rubric.md`; `conf/onnx_sources.json`; `conf/model/object_detection/yolov8/yolov8s.toml`; `conf/server/object_detection/yolov8/yolov8s_server_config.toml`; `conf/packs/`（`demo.toml`、`yolov8.toml`、`metric3d.toml`，并读了三份 pack）；`conf/product_static_bs1.json`；`conf/mortred.toml`；`conf/trt_engines.json`；`conf/trt_engines.set_a.json`；`conf/trt_profiles/yolov8s.json`

## 优势
- 单品交付链路完整：`yolov8s.toml`（`type=tensorrt` + `*.engine.static_bs1.fp16`）与 `yolov8s_server_config.toml`（port/URI/`model_config_file_path`）成对，`product_static_bs1.json` 里 `yolov8s` 条目三者对齐，小团队能按 catalog id 起一个检测服务。
- 互换与契约可查：`onnx_sources.json` 写明 product 后端保持 `mnn`/`tensorrt`（及 LIBFACE 等例外）、IO/preprocess、license、`onnx_status`，出错时能对照图而不是盲改 toml。
- 进程面配置到位：`mortred.toml` 固定 gateway `:8080`、supervisor `:8787`；`conf/packs/*.toml` 用 `[pack.<id>] worker_nums` 做占用层，符合「本机本卡准备、按 pack 拉起」的交付习惯。
- Set A 转换清单与产品命名一致：`trt_engines.set_a.json` 中 yolov8s 的 onnx/engine 与 product toml / `product_static_bs1.json` 同为 `*.static_bs1.fp16`。

## 劣势
- 同一产品 id 的 engine 路径在多份真源里不一致：产品侧是 `yolov8s.engine.static_bs1.fp16`（toml / product_static_bs1 / set_a），`onnx_sources.json` 写成 `...fp16in`，遗留 `trt_engines.json` 仍是 `yolov8s.engine` + 旧 `yolov8s.onnx`；换机 `prepare`/doctor 时极易建错或找错文件。
- 仓库内 pack 过薄且覆盖面窄：仅 `demo` / `yolov8` / `metric3d` 三份，且只有 `worker_nums=1`，没有校准写出的 `gpu_mem_mib` 等；多客户同机并发要靠机外拷贝，conf 本身不够交付就绪。
- conf 树命名与索引分散（`*_config.toml` / 短名、`server_cfg` / `server_config`、两套 trt 清单），小团队接入非 catalog 客户模型时定位成本高；`onnx_sources` 自身也标注部分 toml 参数可能陈旧（如 ATTENTIVE_GAN 尺寸），削弱「改 conf 即真相」的信任。

## 依据
- 分数 6.8：已知 catalog（如 yolov8s）可交付，但 engine 真源分裂 + pack 样本不足，达不到「独立交付且出错必能定位」（rubric 10 分标准）。
- 优势出处：`conf/model/.../yolov8s.toml`、`conf/server/.../yolov8s_server_config.toml`、`conf/product_static_bs1.json`（yolov8s）、`conf/onnx_sources.json`（policy + YOLOV8）、`conf/mortred.toml`、`conf/packs/*.toml`、`conf/trt_engines.set_a.json`。
- 劣势出处：`onnx_sources.json` runtime_path `fp16in` vs toml/`product_static_bs1`/`trt_engines.set_a.json` 的 `fp16` vs `trt_engines.json` 的 `yolov8s.engine`；`conf/packs/` 仅三文件且字段极简；`onnx_sources.json` ATTENTIVE_GAN 对 toml 尺寸 stale 的说明。

## 未读
无
