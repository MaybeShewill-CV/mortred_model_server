# product static HTTP RPS

Catalog: `conf/product_static_bs1.json` (39 product TensorRT ids, JSON order).

This file is a skeleton. It is not PF1 or TP1 evidence. No success RPS,
p50, or p99 is recorded here. Each section stays `status=blocked` and
`evidence=missing` until a GPU host runs the driver and `judge_row` accepts
the row. A blocked section that also contains `success_rps`, `p50_ms`,
`p99_ms`, or `throughput_qps` is rejected.

Policy:

- `--catalog product_static --check` exits non-zero until all 39 sections are qualified PASS rows.
- `--catalog product_static --check-skeleton` exits 0 when the headings match
  JSON order and every section is either explicit blocked/missing or a qualified PASS row.
- Coverage-set nine stay in `docs/perf/coverage_set_http_rps.md` (ops / TP2 subset).
  Those measured rows are not copied here, so this file does not claim PASS for them.

Operator procedure on a GPU host (solo residency; do not require the other ids
to be resident). The driver does not write a PASS row unless `judge_row` accepts it:

1. `mortredctl prepare --id <id>`
2. One-id pack. If `conf/packs/<id>.toml` exists and lists only that id, it is
   the temp pack seed (per-id packs from the pack work). Otherwise the driver
   writes `[pack.<id>] worker_nums = 1`. Git packs are not modified.
3. `mortredctl calibrate --pack <temp> --write-pack --duration 60s`
   (the calibrator posts with `http_infer_rps.py --raw`).
4. Deploy that pack alone (`MORTRED_PACK` is the temp pack).
5. `python3 scripts/server/http_infer_rps.py --raw --model <id>`
   `--gateway http://127.0.0.1:8080 -c <worker_nums×2> -d 60s --warmup 5s`

60s meets the ≥30s bar. Concurrency is the deployed pack `worker_nums` times 2.
`conf/server` stays `worker_nums=1`.

```bash
python3 scripts/coverage_http_report.py --catalog product_static --id yolov8n
python3 scripts/coverage_http_report.py --catalog product_static --from yolov8n
python3 scripts/coverage_http_report.py --catalog product_static --check-skeleton
python3 scripts/coverage_http_report.py --catalog product_static --check
```

A PASS row also writes `docs/perf/baselines/<id>.json` (`schema` `pf5-baseline-v1`)
the first time, and keeps that file when the hardware string matches. No baseline
file is written for a blocked section.

Seed p99 caps for every product id live in `scripts/calibrate_pack.py` `P99_CAP_MS`.
A missing cap aborts calibrate with a table. The latency gate is not skipped.

## yolov8n
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id yolov8n on a machine with nvidia-smi"

## yolov8s
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id yolov8s on a machine with nvidia-smi"

## yolov8l
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id yolov8l on a machine with nvidia-smi"

## yolov8x
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id yolov8x on a machine with nvidia-smi"

## yolov6s
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id yolov6s on a machine with nvidia-smi"

## yolov5l
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id yolov5l on a machine with nvidia-smi"

## yolov7
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id yolov7 on a machine with nvidia-smi"

## yolov7x
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id yolov7x on a machine with nvidia-smi"

## nanodet_1x5
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id nanodet_1x5 on a machine with nvidia-smi"

## nanodet_416
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id nanodet_416 on a machine with nvidia-smi"

## libface
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id libface on a machine with nvidia-smi"

## centerface
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id centerface on a machine with nvidia-smi"

## mobilenetv2
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id mobilenetv2 on a machine with nvidia-smi"

## resnet
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id resnet on a machine with nvidia-smi"

## densenet
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id densenet on a machine with nvidia-smi"

## superpoint
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id superpoint on a machine with nvidia-smi"

## dbnet
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id dbnet on a machine with nvidia-smi"

## pphuman_mobile
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id pphuman_mobile on a machine with nvidia-smi"

## pphuman_lite
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id pphuman_lite on a machine with nvidia-smi"

## pphuman_server
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id pphuman_server on a machine with nvidia-smi"

## bisenetv2
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id bisenetv2 on a machine with nvidia-smi"

## hrnet
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id hrnet on a machine with nvidia-smi"

## msocrnet
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id msocrnet on a machine with nvidia-smi"

## modnet
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id modnet on a machine with nvidia-smi"

## ppmatting_512
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id ppmatting_512 on a machine with nvidia-smi"

## ppmatting_1024
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id ppmatting_1024 on a machine with nvidia-smi"

## ppmatting_resnet34
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id ppmatting_resnet34 on a machine with nvidia-smi"

## ppmatting_v2
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id ppmatting_v2 on a machine with nvidia-smi"

## attentive_gan
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id attentive_gan on a machine with nvidia-smi"

## enlightengan
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id enlightengan on a machine with nvidia-smi"

## realesrgan
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id realesrgan on a machine with nvidia-smi"

## dinov2_vits14
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id dinov2_vits14 on a machine with nvidia-smi"

## dinov2_vitb14
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id dinov2_vitb14 on a machine with nvidia-smi"

## dinov2_vitl14
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id dinov2_vitl14 on a machine with nvidia-smi"

## depth_vits14
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id depth_vits14 on a machine with nvidia-smi"

## depth_vitb14
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id depth_vitb14 on a machine with nvidia-smi"

## depth_vitl14
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id depth_vitl14 on a machine with nvidia-smi"

## metric3d_512
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id metric3d_512 on a machine with nvidia-smi"

## metric3d_1088
status=blocked | evidence=missing | checked=no | pf1=not-pass | reason="no live GPU HTTP RPS; run python3 scripts/coverage_http_report.py --catalog product_static --id metric3d_1088 on a machine with nvidia-smi"
