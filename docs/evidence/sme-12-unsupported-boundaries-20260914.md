# SME-12 evidence — unsupported boundaries docs (review)

- **Date:** 2026-09-14
- **Branch / HEAD:** `fix/sme-12-unsupported-boundaries` @ `c8ca344`
- **Host:** docs review (WSL/Mac); no runtime build required

## Checks

```text
docs/unsupported-boundaries.md
→ Linux x64 only; RTDETR scaffold not in catalog; CUDA 12.x + TensorRT 10.x only;
  engines must rebuild on this GPU/TRT; cpu profile has no TRT

docs/unsupported-boundaries.zh-cn.md
→ zh summary; notes TRT<9 checklist wording superseded by TRT 10 pin

Links present from README (Quick Start + Model Zoo), oob-main-path,
deployment (+ zh), rtdetr model doc
```

## Outcome

**PASS.** Single boundaries page; no false expectations for RTDETR / non-Linux /
TRT 8·9 / copied engines.
