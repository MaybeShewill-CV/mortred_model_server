# Out-of-box main path (SME)

**One lane.** Tokens → listen (loopback) → start → (GPU) pack prepare/calibrate →
`doctor --strict` → one infer. If anything is missing, run:

```bash
mortredctl next
```

It prints exactly one `next:` command. Do that, then `mortredctl next` again.

Default pack: `conf/packs/demo.toml` (`MORTRED_PACK`). Default profile: `cpu`
unless you set `MORTRED_PROFILE=gpu` / `--profile gpu`.

Alternate install shapes (compose / tarball / source build) are **entry variants**
of this same lane — not separate products. See README Quick Start.

What we do **not** support (RTDETR, non-Linux, TRT≠10, copying engines): [unsupported-boundaries.md](unsupported-boundaries.md).

## 0. Get a tree

Pick **one** entry, then continue from §1:

| Entry | How |
|---|---|
| Docker compose | `git clone … && cd mortred_model_server` (weights + compose in §3) |
| Release tarball | unpack + `sudo ./install.sh` → tree at `/opt/mortred` |
| Source (cpu) | `install_deps.sh --cpu --all && --cpu --check` → `cmake --preset full-cpu` → build (details: [shortest-path-cpu.md](shortest-path-cpu.md)) |

## 1. Profile + weights

```bash
mortredctl init --profile cpu    # or omit --profile to auto-detect GPU
# ends with: next: mortredctl next
```

## 2. Three tokens

```bash
mortredctl next                  # -> init-trust if trust.env missing
mortredctl init-trust            # writes conf/local/trust.env (mode 600)
set -a && . conf/local/trust.env && set +a
mortredctl next
```

Do not paste token values into chat. Do not reuse the scrape token as
inference or management.

## 3. Listen (loopback)

Keep gateway `8080` and supervisor `8787` on **127.0.0.1** for the first hour.
Non-loopback plain HTTP → `mortredctl next` points at `init-edge --mode lan`
(or bind loopback). Fail-closed start still requires the three tokens.

## 4. Start supervisor

```bash
mortredctl next                  # -> start command for this tree
# examples:
#   export MORTRED_PACK="$PWD/conf/packs/demo.toml" MORTRED_PROFILE=cpu
#   ./_bin/mortred-supervisor.out
#   # or: docker compose --profile cpu up -d
#   # or: sudo systemctl start mortred-supervisor
curl -fsS http://127.0.0.1:8787/api/v1/health
```

## 5. Pack (GPU / TensorRT only)

cpu + `demo.toml` has no TensorRT backends — `mortredctl next` skips this.

```bash
mortredctl next                  # -> prepare if engines missing
mortredctl prepare --pack "$MORTRED_PACK"
# stop supervisor if ports are busy, then:
mortredctl calibrate --pack "$MORTRED_PACK" --write-pack
mortredctl next
```

## 6. Accept

```bash
mortredctl doctor --strict
```

## 7. One infer

```bash
ROUTE=/mortred_ai_server_v1/classification/mobilenetv2
# build JSON body in a file (do not put large base64 on argv) — see shortest-path-cpu.md §4
curl --max-time 120 -sS \
  -H "Authorization: Bearer ${MORTRED_GATEWAY_AUTH_TOKEN}" \
  -H 'Content-Type: application/json' \
  -X POST "http://127.0.0.1:8080${ROUTE}" \
  --data-binary @/tmp/mortred-infer-body.json
```

## Stuck?

Always: `mortredctl next`. Full ops manual: [deployment.md](deployment.md).
