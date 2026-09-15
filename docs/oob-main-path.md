# Out-of-box main path (SME)

**One lane.** Get a tree → tokens → listen (loopback) → start → (GPU) pack
prepare/calibrate → `doctor --strict` → one infer.

Whenever something is missing:

```bash
mortredctl next
```

It prints exactly one `next:` command. Do that, then run `mortredctl next` again.

Default pack: `conf/packs/demo.toml` (`MORTRED_PACK`). Default profile: `cpu`
unless you set `MORTRED_PROFILE=gpu` / `--profile gpu`.

Docker compose / release tarball / source build are **entry variants of this
same lane** — not separate products. Deep ops (security, upgrades, TRT pack
details): [deployment.md](deployment.md) / [deployment.zh-cn.md](deployment.zh-cn.md).

Unsupported (RTDETR scaffold, non-Linux, TRT ≠ 10.x, copying engines):
[unsupported-boundaries.md](unsupported-boundaries.md).

---

## 0. Get a tree

Pick **one** entry, then continue from §1.

### 0a. Docker compose

```bash
git clone … && cd mortred_model_server
# weights + compose: follow README Quick Start / deployment §5, then §1 below
```

### 0b. Release tarball

Unpack the profile tarball, then `sudo ./install.sh` → tree at `/opt/mortred`.
Continue from §1 inside that tree.

### 0c. Source build (cpu)

Linux only. Use when GPU / `install_deps.sh --all` is unavailable.

```bash
cd /path/to/mortred_model_server
git checkout main && git pull --ff-only origin main
./scripts/install_deps.sh --cpu --all
./scripts/install_deps.sh --cpu --check   # must pass before cmake
```

Configure fail-closed on missing workflow/crypto or mismatched ORT headers
(`ORT_API_VERSION` must match `libonnxruntime.so.1.29.0`); stderr names the fix.

```bash
cmake --preset full-cpu
cmake --build build/full-cpu -j"$(nproc)"
```

GPU source builds use `install_deps.sh --all` / `--check` and `cmake --preset full`
(see deployment §7). After the binary exists, continue from §1 (same lane).

---

## 1. Profile + weights

```bash
mortredctl init --profile cpu    # or omit --profile to auto-detect GPU
# ends with: next: mortredctl next
```

From a raw source tree you can also:

```bash
python3 scripts/fetch_weights.py --profile cpu
```

---

## 2. Three tokens

```bash
mortredctl next                  # -> init-trust if trust.env missing
mortredctl init-trust            # writes conf/local/trust.env (mode 600)
set -a && . conf/local/trust.env && set +a
mortredctl next
```

Do not paste token values into chat. Do not reuse the scrape token as
inference or management.

---

## 3. Listen (loopback)

Keep gateway `8080` and supervisor `8787` on **127.0.0.1** for the first hour.
Non-loopback plain HTTP → `mortredctl next` points at `init-edge --mode lan`
(or bind loopback). Fail-closed start still requires the three tokens.

---

## 4. Start supervisor

```bash
mortredctl next                  # -> start command for this tree
```

Source-tree example:

```bash
export MORTRED_PROFILE=cpu
export MORTRED_PACK="$PWD/conf/packs/demo.toml"
export APP_BIN_DIR="$PWD/_bin" APP_LIB_DIR="$PWD/_lib" APP_LIBS_DIR="$PWD/3rd_party/libs"
export LD_LIBRARY_PATH="$PWD/_lib:$PWD/3rd_party/libs:${LD_LIBRARY_PATH:-}"
pkill -f 'mortred-supervisor.out' 2>/dev/null || true
nohup ./_bin/mortred-supervisor.out > /tmp/mortred-supervisor.log 2>&1 &
sleep 3
curl -fsS http://127.0.0.1:8787/api/v1/health
```

Other shapes: `docker compose --profile cpu up -d`, or
`sudo systemctl start mortred-supervisor`.

---

## 5. Pack (GPU / TensorRT only)

cpu + `demo.toml` has no TensorRT backends — `mortredctl next` skips this.

```bash
mortredctl next                  # -> prepare if engines missing
mortredctl prepare --pack "$MORTRED_PACK"
# stop supervisor if ports are busy, then:
mortredctl calibrate --pack "$MORTRED_PACK" --write-pack
mortredctl next
```

---

## 6. Accept

```bash
mortredctl doctor --strict
```

Caveats:

- `security_warn --self-test` fails if tokens already exported → `env -u` the
  three tokens then re-run.
- densenet 404: live probe greps first `server_uri`, not the demo pack.

---

## 7. One infer

```bash
ROUTE=/mortred_ai_server_v1/classification/mobilenetv2

# unauthenticated should be 401
curl -s -o /dev/null -w 'unauth=%{http_code}\n' -X POST "http://127.0.0.1:8080${ROUTE}"

# put JSON in a file — do not put large base64 on argv
python3 - <<'PY'
import base64, json
from pathlib import Path
p = Path("demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG")
Path("/tmp/mortred-infer-body.json").write_text(
    json.dumps({"images": [base64.b64encode(p.read_bytes()).decode()]})
)
print("body_bytes", Path("/tmp/mortred-infer-body.json").stat().st_size)
PY

curl --max-time 120 -sS -w '\ninfer_http=%{http_code}\n' \
  -H "Authorization: Bearer ${MORTRED_GATEWAY_AUTH_TOKEN}" \
  -H 'Content-Type: application/json' \
  -X POST "http://127.0.0.1:8080${ROUTE}" \
  --data-binary @/tmp/mortred-infer-body.json
# expect body status==0, model MOBILENETV2
```

---

## Stuck?

Always: `mortredctl next`. Full ops manual: [deployment.md](deployment.md).
