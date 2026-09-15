# Model Server Tutorials

Task servers share one binary and one demo client. This page covers **start a
server** and **call it** for the common image tasks.

Prerequisite: a built tree with `_bin/mortred-model-server.out` (see
[oob-main-path.md](oob-main-path.md)).
Config field reference: [about_model_server_configuration.md](about_model_server_configuration.md),
[about_model_configuration.md](about_model_configuration.md).

## Shared: binary and client

```bash
cd $PROJECT_ROOT/_bin
./mortred-model-server.out --model <CATALOG_ID> <path-to-server-toml>
```

Shipped configs default to `worker_nums=1`. Raise it only if you have GPU headroom.

Demo client (stdlib only; no `requests` / locust):

```bash
cd $PROJECT_ROOT
python3 scripts/server/test_server.py --server <alias> --mode single --times 3
```

`scripts/server/test_server.py` also supports closed-loop load modes; see its
`--help`. Server URL comes from the server TOML `port` / URI.

---

## Classification (`MOBILENETV2`)

```bash
./mortred-model-server.out --model MOBILENETV2 \
  ../conf/server/classification/mobilenetv2/mobilenetv2_server_config.toml
```

```bash
python3 scripts/server/test_server.py --server mobilenetv2 --mode single --times 3
```

Screenshots (optional): `resources/images/start_a_mobilenetv2_server.png`,
`mobilenetv2_server_ready.png`, `mobilenetv2_sample_client.png`.

Response: top-k scores under `results[0].data` (see API contract).

---

## Object detection (`YOLOV5`)

```bash
./mortred-model-server.out --model YOLOV5 \
  ../conf/server/object_detection/yolov5/yolov5_server_config.toml
```

Switch yolov5s/m/… via the model TOML ([about_model_configuration.md](about_model_configuration.md)).

```bash
python3 scripts/server/test_server.py --server yolov5 --mode single
```

Each box in `results[0].data`: `class_id`, `score`, `category`, `bbox` `[x1,y1,x2,y2]`.

Screenshot: `resources/images/start_a_yolov5_server.png`.

---

## Scene segmentation (`BISENETV2`)

```bash
./mortred-model-server.out --model BISENETV2 \
  ../conf/server/scene_segmentation/bisenetv2/bisenetv2_server_config.toml
```

```bash
python3 scripts/server/test_server.py --server bisenetv2 --mode single
```

`results[0].data`: `image` (mask PNG base64), `colorized_mask` (colorized PNG base64).

Screenshot: `resources/images/start_a_bisenetv2_server.png`.

---

## Enhancement (`ATTENTIVE_GAN_DERAIN`)

```bash
./mortred-model-server.out --model ATTENTIVE_GAN_DERAIN \
  ../conf/server/enhancement/attentive_gan_derain/attentive_gan_server_cfg.toml
```

```bash
python3 scripts/server/test_server.py --server attentive_gan --mode single
```

One image in `results[0].data.image` (JPEG/PNG base64).

Screenshot: `resources/images/start_a_derain_server.png`.

---

## Feature point (`SUPERPOINT`)

```bash
./mortred-model-server.out --model SUPERPOINT \
  ../conf/server/feature_point/superpoint/superpoint_server_cfg.toml
```

```bash
python3 scripts/server/test_server.py --server superpoint --mode single
```

Points from `fill_feature_points`: each has `score`, `location` `[x,y]`, `descriptor`.

Screenshot: `resources/images/start_a_superpoint_server.png`.

---

## More catalog ids

Full HTTP-served zoo: README **Model Zoo** table (`mortred-model-server.out --list`).
Other tasks (OCR, matting, depth, diffusion, SAM, …) use the same binary pattern:
`--model <ID>` + matching `conf/server/...` TOML + `test_server.py --server <alias>`
when an alias exists.
