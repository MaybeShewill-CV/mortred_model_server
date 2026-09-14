# SME-11 evidence — oob main path + mortredctl next (WSL)

- **Date:** 2026-09-14
- **Branch / HEAD:** `fix/sme-11-oob-main-path` @ `328a296`
- **Host:** WSL (`DESKTOP-B8MJFUS`), `/mnt/g/Codex/mortred_model_server`

## B — trust.env present, tokens unset

```text
env -u MORTRED_GATEWAY_AUTH_TOKEN -u MORTRED_API_TOKEN -u MORTRED_METRICS_TOKEN \
  bash scripts/mortredctl_next.sh
→ why: trust.env exists but the three MORTRED_*_TOKEN vars are not in this shell
→ next: set -a && . …/conf/local/trust.env && set +a
```

## C — tokens sourced, supervisor down

```text
set -a && . conf/local/trust.env && set +a
bash scripts/mortredctl_next.sh
→ why: supervisor not reachable at http://127.0.0.1:8787; start it (source-tree)
→ next: export MORTRED_PACK="…/conf/packs/demo.toml" MORTRED_PROFILE=… && "…/_bin/mortred-supervisor.out"
```

## CLI parity

```text
cmake --build build/full-cpu --target mortredctl.out
./_bin/mortredctl.out next
→ same supervisor-start next as script
```

## Outcome

**PASS** (A optional: move trust.env → init-trust not re-run). Single OOB lane
docs + one-next command behavior verified on WSL.
