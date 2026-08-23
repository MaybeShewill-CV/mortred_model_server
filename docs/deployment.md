# Deployment Guide (Linux)

Mortred targets Linux only. This document is the operational counterpart of
the [Quick Start](../README.md): profiles, distribution tracks, the three
first-hour entries, upgrades and the acceptance gates.

## Profiles: one switch, four layers

`cpu` and `gpu` are not two products - one profile switch drives the build,
the dependency set, the model catalog and the weight subset:

| Layer | Switch | cpu | gpu (default) |
|---|---|---|---|
| Build | `MORTRED_BUILD_PROFILE` | TRT compiled out | full CUDA/TRT stack |
| Dependencies | `install_deps.sh --cpu` | MNN-CPU + ORT-CPU | + CUDA/TRT/cuDNN (root) |
| Catalog | server TOML `profile` field | curated `*_cpu` configs only | everything |
| Weights | `fetch_weights.py --profile` | curated subset | full manifest |

Runtime selection is `MORTRED_PROFILE=cpu|gpu` (default `gpu`); absent
`profile` fields mean `gpu`, so the cpu catalog is always explicitly curated.
The curated cpu model set (mobilenetv2, resnet50, yolov8, hrnet) is frozen
per release - extending it is a CHANGELOG-worthy change, not a config tweak.

## Distribution tracks (equal citizens)

- **Docker**: `docker build --target mortred-cpu` or the default gpu target;
  `docker compose --profile cpu|gpu up -d`. Images are published per release
  as `ghcr.io/<owner>/mortred_model_server:vX.Y.Z-cpu|-gpu`.
- **Tarball**: `scripts/make_release_tarball.sh <profile> <version>`
  produces a self-contained archive (installed tree + systemd unit +
  `install.sh`) with a `.sha256`. `install.sh` wires runtime apt deps,
  `/opt/mortred`, the `mortred` user and the systemd unit. Weights are never
  bundled (tens of GB); `install.sh` prints the matching
  `fetch_weights.py --profile` command.

Both tracks are validated by the same `scripts/verify_deployment.sh`.

## First-hour entries (one core, three shells)

All three end at `mortredctl doctor`:

1. **bootstrap one-liner** (`scripts/bootstrap.sh`): hardware detection
   (nvidia-smi) → docker track or latest release tarball. Thin by design.
2. **docker compose**: profile flag + weight fetch; tokens via environment.
3. **tarball + systemd**: installer, `supervisor.env` tokens, weight fetch,
   `systemctl start`.

`mortredctl init [--profile]` is the programmatic core (detect / fetch /
verify); `mortredctl doctor` runs the live acceptance; `mortredctl upgrade
[version]` performs an in-place upgrade (see below).

## TensorRT engines (gpu only)

Engines are hardware/TRT-version specific artifacts and are never shipped.
Convert the missing ones per machine:

```bash
./scripts/convert_trt_engines.sh --list   # manifest
./scripts/convert_trt_engines.sh          # convert missing (FP16 + profiles)
```

Containers may opt into conversion before autostart with
`MORTRED_AUTO_BUILD_ENGINES=true` (off by default - conversion is
minutes-long). `doctor` warns about missing engines without failing.

## Upgrades

- `mortredctl upgrade` (or `upgrade vX.Y.Z`): downloads the release tarball
  for the RUNNING profile, verifies sha256, backs up `conf/` to
  `conf.backup-<timestamp>`, installs over `/opt/mortred` (weights untouched),
  restarts the service and runs `doctor`.
- In-place upgrades are supported between ADJACENT minor versions. Across
  larger jumps: export configs, fresh-install the target version, migrate
  configs manually (`scripts/migrate_model_config.py` helps).
- Config compatibility breaks are called out in `CHANGELOG.md` per version.

## Acceptance gates

```bash
./scripts/verify_deployment.sh --basic   # static: syntax/manifests/deps
./scripts/verify_deployment.sh --full    # + local weights sha256 + 3rd_party
./scripts/verify_deployment.sh --live    # + gateway probes (healthz/auth)
mortredctl doctor                        # live wrapper
```

CI runs the cpu-profile full build + test suite on a GPU-less runner for
every change - the conditional-compilation path cannot silently rot.
