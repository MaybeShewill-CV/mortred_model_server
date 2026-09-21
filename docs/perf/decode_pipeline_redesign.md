# Decode/Preprocess Pipeline Redesign — Throughput Campaign (2026-09-21)

Branch `perf/decode-pipeline-redesign`. Goal: maximize model-server
throughput under batch=1, approaching the pure-inference GPU floor.

## Design (consensus notes)

- **Two-path fork**: path A (CPU: imdecode-reduced → CPU letterbox → H2D)
  and path B (GPU: jpeggpu → preprocess kernel → TRT, zero host
  round-trips). Decode and preprocess always live on the same side —
  the old mismatched S1 path (GPU decode + CPU preprocess) was
  Pareto-dominated and deleted.
- **Transport-agnostic**: base64 decodes at item birth
  (`bind_parsed_request`); both transports reach the fork as identical
  bytes, cost lands on handler threads (envelope stage).
- **Fork decision**: engine eligibility resolved once at init
  (`resolve_decode_routing`); per request only SOF hard gates
  (progressive/EXIF) + one threshold:
  `est_cpu_ms = cpu_us_per_kb × bytes ≥ gpu_min_cpu_ms`.
  `image_decode_backend` (cpu|auto|gpu) selects the mode; thresholds
  default throughput-oriented (11 µs/KB, 8 ms) pending pack calibration.
- The startup race, nvjpeg ladder, S1 support code and dead hooks are
  deleted (~830 net lines).

## Commits

- `d9a49675` fork refactor (WP1)
- `d311a74d` pinned staging slot pools — fixes a cross-worker data race
  on the shared H2D/D2H buffer that could silently swap tensor bytes
  between in-flight requests (WP3)
- `6341542c` block-wise base64 decode (WP2.2)

## Acceptance matrix (WP4)

C=16, direct :9056, 15s × 3 rounds, medians (spread ±1%):

| case | pre-redesign (gateway, best of era) | post-redesign | vs floor |
|---|---|---|---|
| bus.jpg raw | 164–202 | **337.4** | ≈98.7% of the observed 341.8 ceiling |
| bus.jpg json | 258 | **303.7** | 89% |
| jidu_face 4K raw | 173–202 | **288.6** | 84% |
| jidu_face 4K json | 194–226 | **257.3** | 75% |

Also: C=8 raw (bus) = 334.6 — the pre-redesign best C=16 number at half
the concurrency. Reference note: trtexec is not installed in this WSL;
the ceiling reference is the best observed pure-inference run (341.8).
Path routing verified: auto mode sends bus (est 3.7ms) and jidu
(est 7.5ms) to path A for both transports; force-gpu sends json+bus to
`jpeggpu-zero-copy` (200/200). Output correctness under load: idle vs
mid-load detection payloads byte-identical (3/3).

## Bottleneck findings along the way (WP2.1)

- Neither CPU (6.7/16 cores) nor GPU (73%) saturated at the old 245 rps
  json wall; per-request CPU inflates ~2× under 8-way load (memory
  contention) and the single-stream FIFO leaves feeding gaps.
- Remaining post-redesign gaps: json −10% (envelope+base64 on handler
  threads), 4K −16% (CPU feeding depth).

## Follow-ups (not in this campaign)

1. worker_nums 8→12/16 requires `mortredctl calibrate --write-pack`
   (occupancy gate rejected the uncalibrated change); expected to close
   the 4K feeding gap toward the floor.
2. Pack calibration for the fork thresholds (`cpu_decode_us_per_kb`,
   `decode_gpu_min_cpu_ms`) per the calibration design.
3. Deeper DCT reduction for ≥4MP images (accuracy tradeoff to validate).
4. trtexec install for the formal reference number.
