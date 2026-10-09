# 模块评分

- id: ops
- 权重: 15
- 分数: 7.0
- 读过的路径: `scripts/mortredctl_init.sh`；`scripts/mortredctl_prepare.sh`；`scripts/mortredctl_doctor.sh`；`scripts/prepare_pack.sh`；`scripts/pack_trt.py`；`scripts/product_static_bs1.py`（`check_pack` / `convert_filters_for_pack`）；`scripts/security_warn.sh`；`scripts/verify_deployment.sh`；`docker-compose.yml`；`Dockerfile`；`deploy/`（`mortred-supervisor.service`、`nginx/nginx.conf.skeleton`、`docker-compose.monitoring.yml`，其余为目录清单）；`conf/packs/demo.toml`、`yolov8.toml`、`gpu_yolo_metric3d.toml`、`metric3d.toml`；`conf/model/object_detection/yolov8/yolov8{n,s,l,x}.toml`、`libfacedetection/libface.toml`、`centerface/centerface.toml`；`docs/unsupported-boundaries.md`

## 优势
- `mortredctl init` 按机型选 cpu/gpu。权重拉取失败，或当前 pack 的产品 TensorRT engine 缺失，都非零退出，并打印 `next:`（`scripts/mortredctl_init.sh`）。
- 当前 pack 里的 TensorRT engine 或 occupancy stamp 缺失时，`mortredctl doctor` 在活体检查之前退出 1，并打印 `next: mortredctl prepare --pack <pack>` 与 `next: mortredctl calibrate --pack <pack> --write-pack`（`scripts/mortredctl_doctor.sh`）。`--strict` 另把安全告警变成失败，并打印 `next:`（`scripts/security_warn.sh`）。
- `mortredctl prepare` 只转换当前 pack 的产品 engine，不走裸的 legacy `conf/trt_engines.json`（`scripts/mortredctl_prepare.sh`、`scripts/prepare_pack.sh`）。`conf/packs/yolov8.toml` 与 `conf/packs/gpu_yolo_metric3d.toml` 含 `yolov8s`；选中这两份且 `yolov8s.engine.static_bs1.fp16` 缺失时，doctor 的这条门是非零并带 `next:`。
- `docker-compose.yml` 的 gpu/cpu 默认把 8787/8080 绑在 `127.0.0.1`，并强制三枚 token。`deploy/mortred-supervisor.service` 的宿主单元是 `User=mortred`。`deploy/` 另有 Nginx edge 骨架和本机 loopback 观测栈。

## 劣势
- H4：默认 `mortredctl doctor --strict`（未设 `MORTRED_PACK`，pack 为 `conf/packs/demo.toml`，只有 CPU `mobilenetv2`）在 `yolov8n`、`yolov8s`、`yolov8l`、`yolov8x`、`libface`、`centerface` 的产品 engine 缺失时，TensorRT 段仍打印 `[ok] pack TRT engines present or pack has no TensorRT backends`，不打印 `next: mortredctl prepare`。`--strict` 不改变这道门。demo pack 上 supervisor 健康、三枚 token 为互不相同的 loopback 长 token 时，该命令退出 0。`conf/packs/` 只有 `yolov8s`（以及 `metric3d_512`、`mobilenetv2`）；没有 `[pack.yolov8n]`、`[pack.yolov8l]`、`[pack.yolov8x]`、`[pack.libface]`、`[pack.centerface]`。`mortredctl prepare` 只转换当前 pack 的 engine，因此没有现成命令能为这五个 id 在另一台机器上生成 engine。改完后：这六份 engine 都不存在、且不设置 `MORTRED_PACK` 时，`mortredctl doctor --strict` 退出码非 0，失败输出里有一行 `next:`，执行该行能为这六个 id 在本机生成对应 engine。
- S3：默认发布镜像是 Dockerfile 最后阶段 `mortred-gpu`（`FROM runtime`）。全程没有 `USER`。`docker-compose.yml` 的 `mortred` 与 `mortred-cpu` 没有 `user:`。`docker build -t mortred_model_server:gpu .` 之后，`docker run --rm --entrypoint id mortred_model_server:gpu -u` 打印 `0`。`docs/unsupported-boundaries.md` 没有写必须 root 的原因。改完后：同一条 `id -u` 不再打印 `0`，或该边界文档写明必须 root 的原因。

## 不计毛病
- 训练平台、标注平台、多云平台。
- 一个 catalog id 一个 OS 进程。
- `docs/unsupported-boundaries.md` 里的 zoo-wide 自动构建保持 opt-in，不能把 H4 的 id 移出最高标准。

## 依据
- `scripts/mortredctl_doctor.sh`：默认 pack 为 `conf/packs/demo.toml`；`pack_trt.py --check` 成功时打印 engine 存在或 pack 无 TensorRT；失败时打印 `next: mortredctl prepare --pack $PACK` 后退出 1。`--strict` 只传给 `security_warn.sh`。
- `scripts/pack_trt.py`：`--check` 只检查 pack 里 `type=tensorrt` 的 engine；空列表返回 0。
- `conf/packs/demo.toml` 只有 `[pack.mobilenetv2]`，模型配置是 CPU toml。`yolov8.toml`、`gpu_yolo_metric3d.toml` 含 `[pack.yolov8s]`。四份 pack 都没有 yolov8n/l/x、libface、centerface。
- 产品 toml 均为 `type = "tensorrt"`，engine 分别为 `weights/object_detection/yolov8/yolov8{n,s,l,x}.engine.static_bs1.fp16`、`weights/object_detection/libfacedetection/face_detection_yunet_2026may.engine.static_bs1.fp16`、`weights/object_detection/centerface_detection/centerface.engine.static_bs1.fp16`。
- `scripts/prepare_pack.sh`：无 TensorRT 的 pack 直接退出 0；有则只对 `product_static_bs1.py --convert-filters` 给出的路径调用 `convert_trt_engines.sh`。
- `scripts/security_warn.sh`：loopback 且三枚 token 合格时 `--strict` 退出 0。`scripts/verify_deployment.sh --live` 在 gateway 检查通过时退出 0，doctor 随之退出 0。
- `Dockerfile`：无 `USER`；默认目标为文末 `FROM runtime AS mortred-gpu`。`docker-compose.yml` 的 gpu/cpu 服务无 `user:`。`docs/unsupported-boundaries.md` 无容器必须以 root 运行的原因。宿主单元 `deploy/mortred-supervisor.service` 的 `User=mortred` 不是该默认镜像。

## 未读
无
