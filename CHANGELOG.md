# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/). 每个版本的条目保留英文原文，中文说明
以引用块附于同版本之下。

## [Unreleased]

## [0.1.0] - 2026-08-23

### Added
- Deployment profile system (`cpu` | `gpu`): one switch drives the build
  (`MORTRED_BUILD_PROFILE`), the dependency set (`install_deps.sh --cpu`),
  the model catalog (per-server `profile` field + `MORTRED_PROFILE`) and the
  weight subset (`fetch_weights.py --profile`). The cpu profile compiles
  TensorRT out entirely and ships a curated model set (mobilenetv2, resnet50,
  yolov8, hrnet).
- Dual-track distribution: `mortred-cpu` Docker target + `docker compose
  --profile cpu|gpu`, and versioned binary tarballs
  (`make_release_tarball.sh` + in-tarball `install.sh` with systemd wiring).
- Three first-hour entries sharing one core: `curl | bash` bootstrap, docker
  compose, and `mortredctl init / doctor / upgrade`.
- `MORTRED_AUTO_BUILD_ENGINES=true` opt-in engine conversion at container
  start (gpu profile).
- Project version (`--version`), this changelog, and a tag-driven release
  pipeline building both images and both tarballs.

### Fixed
- `mortred-gateway` link failure against vendored OpenSSL after the P0-2
  rework (no CI path compiled the gateway; the new cpu-profile job now does).

> 中文摘要：新增部署 Profile 体系（cpu/gpu 单一事实源贯穿构建、依赖、目录、
> 权重四层）、双轨分发（Docker 双 target + compose profiles；版本化 tarball
> + systemd 安装器）、三入口共享 mortredctl 内核（bootstrap / compose /
> init-doctor-upgrade）、可选首启 engine 转换、项目版本化与发布流水线；
> 修复 gateway 对 vendored OpenSSL 的链接缺口。
