# Round 12 queue (locked)

> Note: round-011 already held the post-R9 source-only eval, so this fix round uses **round-012**.

Baseline HEAD: `80a394bb` on `sub-agent-co-work`.

## Locked P1s (only these four)

1. **P1-1** waiting_jobs / queue-depth gauges stick after idle
2. **P1-2** Default check / CI TSan structurally fragile
3. **P1-3** EnlightenGAN TU missing model_runtime.h
4. **P1-4** No dual-upstream isolation regression

Packing/Docker OUT OF SCOPE.
