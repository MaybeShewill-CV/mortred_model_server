# 模块评分

- id: ops
- 权重: 15
- 分数: 8.6
- 读过的路径: `scripts/mortredctl_init.sh`；`scripts/mortredctl_prepare.sh`；`scripts/mortredctl_doctor.sh`；`docker-compose.yml`；`deploy/`（含 `mortred-supervisor.service`、`nginx/`、`docker-compose.monitoring.yml`、`prometheus.yml`、`prometheus.compose.yml`、`alert-rules.yml`、`grafana*`）；`scripts/check_legacy_yolov8s_engine_align.py`；对 `scripts/` 与 `.github/` 的 `check_legacy_yolov8s_engine_align` 调用检索

## 优势
- `mortredctl init` 按机型选 cpu/gpu，权重拉取失败与 GPU pack TRT 缺失均 fail-closed，并指向 `mortredctl prepare`（`scripts/mortredctl_init.sh`）。
- `mortredctl doctor` 串联 ASan ELF、pack occupancy、pack TRT、security warn，以及 supervisor 可达时的 live/`--basic` 回退验收（`scripts/mortredctl_doctor.sh`）。
- `mortredctl prepare` 校验 pack 存在后进入 `prepare_pack.sh`，产品引擎走 pack 作用域而非裸 legacy 默认（`scripts/mortredctl_prepare.sh`）。
- `docker-compose.yml` 分 gpu/cpu/edge profile，8787/8080 默认绑 `127.0.0.1`，强制三 token。
- `deploy/` 提供 host systemd、Nginx edge 骨架与本机可观测栈（loopback Prometheus/Grafana），覆盖小团队 Linux 上机路径。

## 劣势
- `scripts/check_legacy_yolov8s_engine_align.py` 已存在且可断言遗留 yolov8s engine/onnx 与 `conf/product_static_bs1.json` 对齐，但 `scripts/mortredctl_doctor.sh`、`scripts/mortredctl_init.sh` 与 `.github/workflows` 均无调用；遗留/产品路径漂移不会被日常 doctor/init 或 CI 拦住。

## 不计毛病
- 训练平台、标注平台、多云平台（按 rubric / `00-rule-change.md`）。

## 依据
- `scripts/mortredctl_init.sh`：profile 探测、`fetch_weights.py`、`pack_trt.py --check`、失败 exit 1。
- `scripts/mortredctl_prepare.sh`：`--pack` 存在性检查后调 `prepare_pack.sh`。
- `scripts/mortredctl_doctor.sh`：ASan / occupancy / TRT / live 门禁；全文无 `check_legacy_yolov8s_engine_align`。
- `scripts/check_legacy_yolov8s_engine_align.py`：对照 `conf/trt_engines.json` 与 `conf/product_static_bs1.json`。
- 仓库内（排除 `cowork/`）对 `check_legacy_yolov8s_engine_align` 无引用；`.github/` 无匹配。
- `docker-compose.yml`、`deploy/mortred-supervisor.service`、`deploy/nginx/*`、`deploy/docker-compose.monitoring.yml`：交付与观测骨架。

## 未读
无
