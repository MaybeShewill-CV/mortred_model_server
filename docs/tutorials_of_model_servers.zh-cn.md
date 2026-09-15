# 模型服务教程（合并版）

各任务共用一个二进制和一个演示客户端。本文说明如何**启动服务**并**发起调用**。

前置：已编译出 `_bin/mortred-model-server.out`（见
[deployment.zh-cn.md](deployment.zh-cn.md)）。
配置字段：[about_model_server_configuration.zh-cn.md](about_model_server_configuration.zh-cn.md)、
[about_model_configuration.zh-cn.md](about_model_configuration.zh-cn.md)。

## 共用：二进制与客户端

```bash
cd $PROJECT_ROOT/_bin
./mortred-model-server.out --model <CATALOG_ID> <path-to-server-toml>
```

仓库自带配置默认 `worker_nums=1`，仅在显存足够时再加大。

演示客户端（标准库即可）：

```bash
cd $PROJECT_ROOT
python3 scripts/server/test_server.py --server <alias> --mode single --times 3
```

服务 URL 来自 server TOML 的 `port` / URI。HTTP 契约见
[api-contract.zh-cn.md](api-contract.zh-cn.md)。

---

## 分类（`MOBILENETV2`）

```bash
./mortred-model-server.out --model MOBILENETV2 \
  ../conf/server/classification/mobilenetv2/mobilenetv2_server_config.toml
```

```bash
python3 scripts/server/test_server.py --server mobilenetv2 --mode single --times 3
```

示意截图：`resources/images/start_a_mobilenetv2_server.png` 等。

---

## 目标检测（`YOLOV5`）

```bash
./mortred-model-server.out --model YOLOV5 \
  ../conf/server/object_detection/yolov5/yolov5_server_config.toml
```

切换 yolov5s/m/… 改模型 TOML（[about_model_configuration.zh-cn.md](about_model_configuration.zh-cn.md)）。

```bash
python3 scripts/server/test_server.py --server yolov5 --mode single
```

`results[0].data` 中每框：`class_id`、`score`、`category`、`bbox` `[x1,y1,x2,y2]`。

---

## 场景分割（`BISENETV2`）

```bash
./mortred-model-server.out --model BISENETV2 \
  ../conf/server/scene_segmentation/bisenetv2/bisenetv2_server_config.toml
```

```bash
python3 scripts/server/test_server.py --server bisenetv2 --mode single
```

`results[0].data`：`image`（mask PNG base64）、`colorized_mask`（上色 PNG base64）。

---

## 图像增强（`ATTENTIVE_GAN_DERAIN`）

```bash
./mortred-model-server.out --model ATTENTIVE_GAN_DERAIN \
  ../conf/server/enhancement/attentive_gan_derain/attentive_gan_server_cfg.toml
```

```bash
python3 scripts/server/test_server.py --server attentive_gan --mode single
```

`results[0].data.image` 为一张 JPEG/PNG base64 图。

---

## 特征点（`SUPERPOINT`）

```bash
./mortred-model-server.out --model SUPERPOINT \
  ../conf/server/feature_point/superpoint/superpoint_server_cfg.toml
```

```bash
python3 scripts/server/test_server.py --server superpoint --mode single
```

每点含 `score`、`location` `[x,y]`、`descriptor`（见 `fill_feature_points`）。

---

## 更多 catalog id

完整列表见 README **Model Zoo**（`mortred-model-server.out --list`）。其它任务
（OCR、抠图、深度、扩散、SAM 等）同一模式：`--model <ID>` + 对应
`conf/server/...` TOML；有 alias 时用 `test_server.py --server <alias>`。
