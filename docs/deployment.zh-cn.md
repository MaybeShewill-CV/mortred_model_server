# 部署指南（Linux）

Mortred 仅面向 Linux。本文是 [快速开始](../README.md) 的运维对应篇：Profile、
双轨分发、三个首小时入口、升级与验收门禁。

## Profile：一个开关，四层贯穿

`cpu` 与 `gpu` 不是两个产品——同一个 profile 开关驱动构建、依赖集、模型
目录与权重子集：

| 层 | 开关 | cpu | gpu（默认） |
|---|---|---|---|
| 构建 | `MORTRED_BUILD_PROFILE` | TRT 编译排除 | 完整 CUDA/TRT 栈 |
| 依赖 | `install_deps.sh --cpu` | MNN-CPU + ORT-CPU | 另加 CUDA/TRT/cuDNN（root） |
| 目录 | server TOML `profile` 字段 | 仅精选 `*_cpu` 配置 | 全部 |
| 权重 | `fetch_weights.py --profile` | 精选子集 | 完整 manifest |

运行时选择用 `MORTRED_PROFILE=cpu|gpu`（默认 `gpu`）；缺失 `profile` 字段按
`gpu` 处理，因此 cpu 目录永远显式精选。精选 cpu 模型集（mobilenetv2、
resnet50、yolov8、hrnet）随版本冻结——扩充是 CHANGELOG 级变更而非配置微调。

## 分发双轨（同等公民）

- **Docker**：`docker build --target mortred-cpu` 或默认 gpu target；
  `docker compose --profile cpu|gpu up -d`。镜像随每次发布推到
  `ghcr.io/<owner>/mortred_model_server:vX.Y.Z-cpu|-gpu`。
- **Tarball**：`scripts/make_release_tarball.sh <profile> <version>` 产出自
  包含归档（安装树 + systemd 单元 + `install.sh`）附 `.sha256`。
  `install.sh` 接线运行时 apt 依赖、`/opt/mortred`、`mortred` 用户与
  systemd 单元。权重永不打包（数十 GB）；`install.sh` 会打印对应的
  `fetch_weights.py --profile` 命令。

两条轨道由同一个 `scripts/verify_deployment.sh` 验收。

## 首小时三入口（一个内核，三个壳）

三者殊途同归于 `mortredctl doctor`：

1. **bootstrap 一行命令**（`scripts/bootstrap.sh`）：硬件检测（nvidia-smi）
   → docker 轨或最新 release tarball。刻意保持薄。
2. **docker compose**：profile 旗标 + 权重拉取；token 走环境变量。
3. **tarball + systemd**：安装器、`supervisor.env` token、权重拉取、
   `systemctl start`。

`mortredctl init [--profile]` 是程序化内核（检测/拉取/验证）；
`mortredctl doctor` 跑实时验收；`mortredctl upgrade [version]` 执行原地升级。

## TensorRT engine（仅 gpu）

engine 是硬件/TRT 版本相关工件，永不随包分发。按机器转换缺失项：

```bash
./scripts/convert_trt_engines.sh --list   # 清单
./scripts/convert_trt_engines.sh          # 转换缺失项（FP16 + profiles）
```

容器可用 `MORTRED_AUTO_BUILD_ENGINES=true` 选择在 autostart 前转换（默认
关闭——转换耗时分钟级）。`doctor` 对缺失 engine 告警但不判失败。

## 升级

- `mortredctl upgrade`（或 `upgrade vX.Y.Z`）：下载当前 profile 的 release
  tarball，校验 sha256，备份 `conf/` 到 `conf.backup-<timestamp>`，覆盖安装
  `/opt/mortred`（权重不动），重启服务并跑 `doctor`。
- 原地升级仅支持相邻 minor 版本。跨更大版本：导出配置、全新安装目标版本、
  手工迁移配置（`scripts/migrate_model_config.py` 可辅助）。
- 配置兼容性破坏逐版本记录于 `CHANGELOG.md`。

## 验收门禁

```bash
./scripts/verify_deployment.sh --basic   # 静态：语法/manifest/依赖
./scripts/verify_deployment.sh --full    # + 本地权重 sha256 + 3rd_party
./scripts/verify_deployment.sh --live    # + 网关探活（healthz/鉴权）
mortredctl doctor                        # live 封装
```

CI 在无 GPU runner 上对每次变更运行 cpu-profile 全量构建 + 测试——条件编译
路径不会悄然腐烂。
