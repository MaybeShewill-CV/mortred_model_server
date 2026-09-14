# Unsupported boundaries (no false expectations)

Mortred is sold and documented for a **narrow, supported envelope**. This page
is the single list of what is **out of scope**. Source and CMake are ground
truth; CHANGELOG is reference-only.

For the first-hour lane, see [oob-main-path.md](oob-main-path.md). Full ops:
[deployment.md](deployment.md).

## Operating system

| Supported | Not supported |
|---|---|
| Linux x64 (Ubuntu 20.04 / 22.04 documented) | Windows, macOS, other CPU ISAs as a product track |

Build scripts, packs, `mortredctl`, and release tarballs assume Linux. Do not
expect a supported Windows/macOS binary line.

## Models / catalog

| Item | Status |
|---|---|
| HTTP catalog (`mortred-model-server.out --list`) | Served ids in [README Model Zoo](../README.md#model-zoo) |
| **RTDETR** | **Scaffold only** — generated skeleton, returns `MODEL_NOT_IMPLEMENTED`, **not registered** in the factory / HTTP catalog. See [models/object_detection/rtdetr.md](models/object_detection/rtdetr.md). Do not treat as a shipped detector. |
| MOT | No MOT HTTP service |
| Bench-only ids | Listed under README “Bench-only”; not the SME HTTP path |

## GPU stack (gpu profile)

| Supported | Not supported |
|---|---|
| CUDA **12.x** | CUDA 11 / 13+ as the product line |
| TensorRT **10.x** (`libnvinfer.so.10`, headers under `TensorRT-10*`) | TensorRT **8** and **9** (leftover `libnvinfer.so.8*` fails configure; pin is 10.3) |
| ORT **1.29** with matching headers (`ORT_API_VERSION==29`) | Mixing ORT 1.18 headers with 1.29 libs (configure fail-closed; see SME-10) |

CMake and `install_deps.sh` refuse leftover TRT 8 / mismatched ORT. There is
**no** supported “TRT 9 halfway” path — use TRT 10 or the **cpu** profile.

## TensorRT engines

Engines are bound to **this GPU + this TensorRT build**. Unsupported as an
out-of-box path:

- Copying `.engine` files from another machine / another TRT major
- Assuming a release tarball’s engines run on your card without convert
- Skipping `mortredctl prepare --pack …` when the pack lists TensorRT backends

Supported path: convert on the target machine for the current pack
(`mortredctl prepare`, deployment §10). Zoo-wide auto-build stays opt-in.

## cpu profile

| Supported | Not supported on cpu |
|---|---|
| MNN-CPU + ORT-CPU | Compiling / linking TensorRT into the cpu preset |
| Demo pack without TRT backends | Expecting `type=tensorrt` model configs to run (init fails clearly) |

## What this page is not

- Not a feature roadmap (RTDETR stays scaffold until registered).
- Not a substitute for `mortredctl doctor --strict` or `install_deps --check`.

---

## 不支持边界（摘要）

- **仅 Linux x64**；无 Windows / macOS 产品线。
- **RTDETR**：脚手架，未进 HTTP catalog，勿当已上线检测器。
- **GPU**：CUDA 12.x + **TensorRT 10.x only**（8/9 不支持）。
- **Engine**：须在本机本卡用当前 TRT **重建**；禁止跨机拷贝当开箱路径。
- **cpu profile**：无 TRT；`tensorrt` 配置会失败。
