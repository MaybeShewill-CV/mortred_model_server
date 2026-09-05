# Repository Layout

This document defines the canonical repository layout and the mapping between source files,
generated executables, configuration files, and documentation.

## Top-level directories

| Path | Purpose | Tracked in VCS |
|---|---|---|
| `src/` | C++ source code (common / models / server / factory / apps) | Yes |
| `conf/` | Model/server configuration, autostart packs (`conf/packs/`), weight manifest, hosted golden CI contract (`ci_hosted_golden.json`) | Yes |
| `docs/` | User and developer documentation | Yes |
| `demo_data/` | Small sample images used by tutorials/benchmarks | Yes |
| `test/` | Unit tests and golden test data | Yes |
| `scripts/` | Build/dependency/test helper scripts | Yes |
| `templates/` | Source templates consumed by `scripts/new_model.py` | Yes |
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
│   ├── common/              # shared app helpers: model_server_main, benchmark_runner
│   ├── benchmark/           # family benchmark drivers used by the unified bench CLI
│   ├── control/             # thin mains for gateway / supervisor / mortredctl
│   ├── product_index.*      # catalog projection for --model
│   ├── model_server_main.cpp
│   └── model_benchmark_main.cpp
├── common/                  # shared utility library: base64, cv_utils, auth, parser,
│                            # process_stop (header-only, workflow daemons only)
├── factory/                 # model/server type-erased factory and registration headers
├── models/                  # model inference implementations
└── server/                  # reusable HTTP server framework (BaseAiServerImpl, AsyncJobTable)
```

## Executable to source mapping

Only executables that can be built from the current source tree are part of the supported
repository layout. Any other executable found under `_bin/` is considered a stale artifact
and must not be relied upon.

### Server executables

| Executable | Source | Server config directory |
|---|---|---|
| `mortred-supervisor.out` | `src/apps/control/supervisor.cpp` (impl: `src/control/`) | environment / conf/mortred.toml |
| `mortred-gateway.out` | `src/apps/control/gateway.cpp` (impl: `src/control/`) | conf/mortred.toml |
| `mortredctl.out` | `src/apps/control/mortredctl.cpp` (impl: `src/control/`) | — |
| `mortred-model-server.out` | `src/apps/model_server_main.cpp` | `conf/server/<task>/<model>/` (`--model <ID>`) |

Identity is the factory catalog `model_section` (`YOLOV8`, `MOBILENETV2`, …).
`mortred-model-server.out --list` prints the HTTP-capable ids.
The supervisor autostart set is `conf/packs/demo.toml` (or `MORTRED_PACK`), not the whole `conf/server/` tree.
Pack TensorRT engines are converted with `scripts/prepare_pack.sh` (`mortredctl prepare`); the supervisor will not spawn a TRT id whose engine file is missing or empty.
Worker_nums calibration is `scripts/calibrate_pack.py` (`mortredctl calibrate`): JSON report, optional `--write-pack` updates the pack file only, `conf/server` stays `worker_nums=1`. GPU occupancy is NVML per-process (or a pre-spawn device delta on WSL), not whole-card `memory.used`.

### Benchmark/tool executables

| Executable | Source |
|---|---|
| `mortred-model-benchmark.out` | `src/apps/model_benchmark_main.cpp` |
| `mortred-supervisor.out` / `mortred-gateway.out` / `mortredctl.out` | `src/apps/control/` (impl: `src/control/`) |

> ONNX→TensorRT 引擎转换不再内置自研转换器，统一由外部 `trtexec`（TensorRT 官方 CLI）
> 执行，驱动脚本为 `scripts/convert_trt_engines.sh`；`trtexec` 由
> `scripts/install_deps.sh` 的 `--nvidia` 模式安装到 `3rd_party/bin/`。

## Naming conventions

- Model IO contracts live in `src/models/io/`, one header per task; `src/models/model_io_define.h` is a compatibility aggregate kept as a pure
  include list. New code should include the task header it needs.
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
Every server config should declare `model = "<catalog id>"` and
`server_exe = "mortred-model-server.out"` so the supervisor/gateway catalog
(`Catalog` in `src/control/`) can spawn `mortred-model-server.out --model <id>`.

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
