# Repository Layout

This document defines the canonical repository layout and the mapping between source files,
generated executables, configuration files, and documentation.

## Top-level directories

| Path | Purpose | Tracked in VCS |
|---|---|---|
| `src/` | C++ source code (common / models / server / factory / apps) | Yes |
| `conf/` | Model and server configuration examples | Yes |
| `docs/` | User and developer documentation | Yes |
| `demo_data/` | Small sample images used by tutorials/benchmarks | Yes |
| `test/` | Unit tests and golden test data | Yes |
| `scripts/` | Build/dependency/test helper scripts | Yes |
| `resources/` | README images and static assets | Yes |
| `3rd_party/` | Vendored third-party headers/libs, populated by `scripts/setup_full_deps.sh` | No (generated/fetched) |
| `_bin/` | CMake executable output directory | No (ignored) |
| `_lib/` | CMake shared-library output directory | No (ignored) |
| `build*/` | CMake build directories | No (ignored) |
| `logs/` | Runtime logs | No (ignored) |
| `generated_configs/` | legacy generated server configs | No (ignored) |
| `weights/` | Large model weight files downloaded by users | No (ignored) |

## Source tree layout

```text
src/
├── apps/
│   ├── common/              # shared app entry points: model_server_main, benchmark_runner
│   ├── model_benchmark/     # per-model benchmark executables
│   ├── server/              # model server executables
│   ├── supervisor/          # control-plane daemon (REST API + embedded UI)
│   ├── gateway/             # data-plane reverse proxy
│   └── cli/                 # mortredctl operations CLI
├── common/                  # shared utility library: base64, cv_utils, auth, parser...
├── factory/                 # model/server type-erased factory and registration headers
├── models/                  # model inference implementations
└── server/                  # reusable HTTP server framework (BaseAiServerImpl)
```

## Executable to source mapping

Only executables that can be built from the current source tree are part of the supported
repository layout. Any other executable found under `_bin/` is considered a stale artifact
and must not be relied upon.

### Server executables

| Executable | Source | Server config directory |
|---|---|---|
| `mortred-supervisor` | `src/apps/control/supervisor.cpp` (impl: `src/control/`) | environment / conf/mortred.toml |
| `resnet_classification_server.out` | `src/apps/server/classification/resnet_classification_server.cpp` | `conf/server/classification/resnet/` |
| `mobilenetv2_classification_server.out` | `src/apps/server/classification/mobilenetv2_classification_server.cpp` | `conf/server/classification/mobilenetv2/` |
| `densenet_classification_server.out` | `src/apps/server/classification/densenet_classification_server.cpp` | `conf/server/classification/densenet/` |
| `yolov5_detection_server.out` | `src/apps/server/object_detection/yolov5_detection_server.cpp` | `conf/server/object_detection/yolov5/` |
| `yolov6_detection_server.out` | `src/apps/server/object_detection/yolov6_detection_server.cpp` | `conf/server/object_detection/yolov6/` |
| `yolov7_detection_server.out` | `src/apps/server/object_detection/yolov7_detection_server.cpp` | `conf/server/object_detection/yolov7/` |
| `yolov8_detection_server.out` | `src/apps/server/object_detection/yolov8_detection_server.cpp` | `conf/server/object_detection/yolov8/` |
| `nanodet_detection_server.out` | `src/apps/server/object_detection/nanodet_detection_server.cpp` | `conf/server/object_detection/nano_det/` |
| `centerface_detection_server.out` | `src/apps/server/object_detection/centerface_detection_server.cpp` | `conf/server/object_detection/center_face_det/` |
| `libface_detection_server.out` | `src/apps/server/object_detection/libface_detection_server.cpp` | `conf/server/object_detection/libface_det/` |
| `bisenetv2_segmentation_server.out` | `src/apps/server/scene_segmentation/bisenetv2_segmentation_server.cpp` | `conf/server/scene_segmentation/` |
| `hrnet_segmentation_server.out` | `src/apps/server/scene_segmentation/hrnet_segmentation_server.cpp` | `conf/server/scene_segmentation/` |
| `pphuman_segmentation_server.out` | `src/apps/server/scene_segmentation/pphuman_segmentation_server.cpp` | `conf/server/scene_segmentation/` |
| `modnet_server.out` | `src/apps/server/matting/modnet_server.cpp` | `conf/server/matting/` |
| `pp_matting_server.out` | `src/apps/server/matting/pp_matting_server.cpp` | `conf/server/matting/` |
| `attentive_gan_derain_server.out` | `src/apps/server/enhancement/attentive_gan_derain_server.cpp` | `conf/server/enhancement/` |
| `enlighten_gan_server.out` | `src/apps/server/enhancement/enlighten_gan_server.cpp` | `conf/server/enhancement/` |
| `real_esrgan_server.out` | `src/apps/server/enhancement/real_esrgan_server.cpp` | `conf/server/enhancement/` |
| `superpoint_fp_det_server.out` | `src/apps/server/feature_point/superpoint_fp_det_server.cpp` | `conf/server/feature_point/` |
| `depth_anything_estimation_server.out` | `src/apps/server/mono_depth_estimation/depth_anything_estimation_server.cpp` | `conf/server/mono_depth_estimation/` |
| `metric3d_estimation_server.out` | `src/apps/server/mono_depth_estimation/metric3d_estimation_server.cpp` | `conf/server/mono_depth_estimation/` |
| `dbnet_text_region_detect_server.out` | `src/apps/server/ocr/dbnet_text_region_detect_server.cpp` | `conf/server/ocr/` |

### Benchmark/tool executables

| Executable group | Source |
|---|---|
| `*_benchmark.out` | `src/apps/model_benchmark/` |
| `mortred-supervisor` / `mortred-gateway` / `mortredctl` | `src/apps/control/` (impl: `src/control/`) |

> ONNX→TensorRT 引擎转换不再内置自研转换器，统一由外部 `trtexec`（TensorRT 官方 CLI）
> 执行，驱动脚本为 `scripts/convert_trt_engines.sh`；`trtexec` 由
> `scripts/install_deps.sh` 的 `--nvidia` 模式安装到 `3rd_party/bin/`。

## Naming conventions

- The canonical diffusion model directory is `src/models/diffusion/`. The historical
  misspelled directory name (diffussion) has been removed; all code should use
  `src/models/diffusion/`.
- Configuration files use TOML syntax. The canonical extension is `.toml`. Historical
  `.ini` files have been migrated; no new `.ini` files should be added.
- Server URI fields use `server_uri` consistently; `server_url` is accepted only for backward
  compatibility during migration.

## Configuration layout

```text
conf/
├── model/       # model inference configuration (TOML syntax)
└── server/      # model server configuration
```

Every server config should reference a model config through `model_config_file_path`.
Every server config should be discoverable from the matching server executable via the
supervisor/gateway catalog (`Catalog` in `src/control/`).

## Stale artifacts policy

The following files are **not** part of the current repository layout. If they appear in
`_bin/`, they should be deleted or moved out of the source tree:

- `llama3_chatbot_server.out`
- `qwen2_vl_chatbot_server.out`
- `ollama_to_llama_cpp_proxy_server.out`
- `jina_embedding_v3_benchmark.out`
- `build_wiki_corpus_index.out`
- `search_wiki_corpus.out`
- `tokenizer_benchmark.out`
- `llm_request_parser_unittest`
- `llm_datatype_unittest`

Their source code is not present in this repository snapshot. Keeping binaries without
source creates an unmaintainable repository and violates the consistency policy.

The proxy server is **not implemented** in this repository; the placeholder docs
about_proxy_server_configuration (English and Chinese variants) were removed. Do not
re-add them until an actual proxy server exists.

## Consistency checks

Before committing, run:

```bash
# Remove generated/build/runtime artifacts from a local working tree
./scripts/clean_artifacts.sh

# Ensure no build artifacts are tracked/left in expected source paths
./scripts/check_repo_clean.sh

# Ensure source/config/docs references are consistent
python3 scripts/check_consistency.py
```

These scripts are part of the repository hygiene CI gate.
