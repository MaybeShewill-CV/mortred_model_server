# ONNX 自导出指南（for models without prebuilt ONNX）

本文档给出"HF 无现成 onnx、需自导出"模型的源权重 → onnx 的标准导出命令。
源权重由 `scripts/download_source_weights.ps1` 下载到 `weights_src/`。
导出后的 onnx 统一放到 `weights/<模型路径>/` 并重新生成 manifest。

> 前置：Python 3.8+、`pip install torch onnx onnxruntime`（Paddle 系还需
> `paddlepaddle paddle2onnx`；TF 系还需 `tensorflow tf2onnx`）。

## 1. yolov6 (n/s/m/l)

源: `weights_src/yolov6/yolov6{n,s,m,l}.pt`（Meituan YOLOv6 v3.0）

```bash
git clone https://github.com/meituan/YOLOv6 && cd YOLOv6
pip install -r requirements.txt
# 逐个导出
python tools/export_onnx.py --weights ../weights_src/yolov6/yolov6n.pt --batch 1 --img-size 640
python tools/export_onnx.py --weights ../weights_src/yolov6/yolov6s.pt --batch 1 --img-size 640
python tools/export_onnx.py --weights ../weights_src/yolov6/yolov6m.pt --batch 1 --img-size 640
python tools/export_onnx.py --weights ../weights_src/yolov6/yolov6l.pt --batch 1 --img-size 640
# 产物拷到 weights/object_detection/yolov6/yolov6{n,s,m,l}.onnx
```

注意：v3.0 的导出默认含 NMS 与否取决于参数（`--end2end`），项目代码期望
输入 `images`、输出 `outputs`（yolov6_detector.inl），导出后需核对节点名，
必要时用 onnx 改名脚本对齐。

## 2. nanodet-plus-m (1.5x/1x)

源: `weights_src/nanodet/nanodet-plus-m_*.pth`（RangiLyu/nanodet）

```bash
git clone https://github.com/RangiLyu/nanodet && cd nanodet
pip install -r requirements.txt
python demo/export_onnx.py --cfg config/ssd/mscoco_nanodet-plus-m_1.5x_416.yml \
  --checkpoint ../weights_src/nanodet/nanodet-plus-m_1.5x_coco.pth
```

注意：nanodet-plus 的 onnx 无 NMS，输出三个 stride 的 head（cls/dis/box），
项目 nano_detector.inl 期望输入 `data`、输出 `output`——GFL v2 解码与节点名
需在导出后核对/对齐（参考 FastDeploy nanodet_plus 的预处理后处理实现）。

## 3. superpoint (5 档分辨率)

源: `weights_src/superpoint/superpoint_v1.pth`（官方 MagicLeap）

```bash
# 参考 fabio-sim/LightGlue-ONNX 的 superpoint.py 导出脚本
python export_superpoint.py \
  --weights ../weights_src/superpoint/superpoint_v1.pth \
  --input-size 120 160   # 分别用 120x160 / 240x320 / 400x800 / 480x640 / 960x1280
```

导出约束：输入节点名 `input`（1x1xHxW），输出 `output_1`（semi）/`output_2`
（desc），与 superpoint.inl 的 `{"input"} -> {"output_1","output_2"}` 完全对应。

## 4. bisenetv2 (cityscapes)

源: PaddleSeg 预训练（bisenetv2_cityscapes，见脚本清单）

```bash
git clone https://github.com/PaddlePaddle/PaddleSeg && cd PaddleSeg
python export.py --config configs/bisenetv2/bisenetv2_cityscapes_1024x1024_160k.yml \
  --model_path ../weights_src/bisenetv2/model.pdparams \
  --save_dir ./export
# 静态图导出后转 onnx
paddle2onnx --model_dir ./export --model_filename model.pdmodel \
  --params_filename model.pdiparams --save_file ../weights_src/bisenetv2/bisenetv2_cityscapes.onnx \
  --opset_version 11 --enable_onnx_checker
```

注意：项目 bisenetv2.inl 期望输入 `input_tensor`、输出 `final_output`（MNN 命名），
paddle2onnx 产物节点名不同，需用 onnx 改名脚本对齐（node 改名 + 输入输出名）。

## 5. pphumanseg (lite 192x192 / server 512x512)

源: PaddleSeg contrib/PP-HumanSeg 预训练（见脚本清单）

```bash
# lite
git clone https://github.com/PaddlePaddle/PaddleSeg && cd PaddleSeg/contrib/PP-HumanSeg
python export.py --config configs/portrait_pp_humansegv2_lite.yml \
  --model_path ../weights_src/pphumanseg/pp_humansegv2_lite.pdparams --save_dir ./export_lite
paddle2onnx --model_dir ./export_lite ... --save_file pp_humansegv2_lite.onnx
# server 同理（portrait_pp_humansegv2_server.yml, 512x512）
```

注意：项目 pp_humanseg.inl 期望输入 `x`、输出 `softmax_0.tmp_0`——paddle2onnx
产物节点名不同，需对齐。

## 6. modnet (hrnet_w18)

源: PaddleSeg PP-ModNet 或 lijiacai/pp-matting 的 modnet-hrnet_w18.pdparams

```bash
# 若来自 PaddleSeg PP-ModNet
python export.py --config configs/ports/portraitnet/pp_modnet_hrnet_w18.yml \
  --model_path ../weights_src/modnet/modnet-hrnet_w18.pdparams --save_dir ./export
paddle2onnx ... --save_file modnet-hrnet_w18.onnx
```

注意：项目 modnet_matting.inl 期望输入 `img`、输出 `sigmoid_2.tmp_0`。

## 7. ppmatting (hrnet_w18 / resnet34_vd)

源: lijiacai/pp-matting 的 pdparams

```bash
# 用 PaddleSeg PP-Matting 配置导出（ppmatting_hrnet_w18 / ppmatting_resnet34_vd）
python export.py --config configs/matting/ppmatting/ppmatting_hrnet_w18.yml \
  --model_path ../weights_src/ppmatting/ppmatting-hrnet_w18.pdparams --save_dir ./export
paddle2onnx ... --save_file ppmatting-hrnet_w18.onnx
```

## 8. enlightengan

源: soralire/EnlightenGAN 官方 checkpoint（需手动从 Google Drive 下载，脚本标注）

```bash
git clone https://github.com/soralire/EnlightenGAN && cd EnlightenGAN
# 加载官方 generator checkpoint，U-Net 结构 1x3xHxW -> 1x3xHxW
python export_onnx.py --checkpoint ../weights_src/enlightengan/Epoch_latest.pth \
  --input-size 1 3 H W   # 按项目 enlightengan.inl 期望的输入尺寸
```

注意：项目 enlightengan.inl 期望双输入 `input_src`/`input_gray`、输出 `output`，
导出时需把 generator 包装成双输入接口（或用两个 onnx）。

## 9. attentive_gan_derain

源: MaybeShewill-CV/attentive-gan-derainnet TF checkpoint（或官方 rui1996/DeRaindrop）

```bash
# TF1 -> frozen pb -> onnx
python freeze_graph.py ...   # 冻结为 .pb
python -m tf2onnx.convert --input frozen_graph.pb --inputs input_tensor:0 \
  --outputs final_output:0 --output attentive_gan_derain.onnx --opset 11
```

注意：项目 attentive_gan_derain_net.inl 期望输入 `input_tensor`、输出 `final_output`
（正好与 TF 命名一致，tf2onnx 后需核对）。输入为 NHWC 1xHxWx3，需转 NCHW。

## 10. lightglue（可选：不下载源权重，直接用现有 onnx）

本地 `weights/feature_point/lightglue/` 已有 `extractor.onnx`、`matcher.onnx`、
`superpoint_lightglue_end2end.onnx` —— **无需下载/导出**，直接作为 HF 基础文件。

---

## 通用后处理：节点名对齐

绝大多数导出的 onnx 节点名与项目代码硬编码不同。统一做法（配合后端抽象方案）：

```bash
pip install onnx
python - <<'PY'
import onnx
m = onnx.load("model.onnx")
# 1) 改输入输出名
m.graph.input[0].name = "input_tensor"
m.graph.output[0].name = "output_tensor"
# 2) 改中间节点名（若代码引用中间节点）
for n in m.graph.node:
    if n.name == "old_mid": n.name = "new_mid"
    n.input[:] = ["input_tensor" if i == "old_in" else i for i in n.input]
    n.output[:] = ["output_tensor" if o == "old_out" else o for o in n.output]
onnx.checker.check_model(m)
onnx.save(m, "model_renamed.onnx")
PY
```

最终节点名以各模型 .inl 中的 `_m_net.init(cfg, {"输入"}, {"输出"})` 为准。
