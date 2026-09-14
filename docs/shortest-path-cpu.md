# Shortest success path (cpu, build from source)

Linux only. Ends at one authenticated inference through the gateway.
Use when GPU / `install_deps.sh --all` is unavailable.

## 0. Repo + deps

```bash
cd /path/to/mortred_model_server
git checkout main && git pull --ff-only origin main
# Clean machine: ./scripts/install_deps.sh --cpu --all
```

## 1. Build

```bash
cmake --preset full-cpu
cmake --build build/full-cpu -j"$(nproc)"
```

## 2. Trust

```bash
./scripts/mortredctl_init-trust.sh --force
set -a && . conf/local/trust.env && set +a
```

Do not paste token values into chat.

## 3. Weights + start

```bash
python3 scripts/fetch_weights.py --profile cpu

export MORTRED_PROFILE=cpu
export MORTRED_PACK="$PWD/conf/packs/demo.toml"
export APP_BIN_DIR="$PWD/_bin" APP_LIB_DIR="$PWD/_lib" APP_LIBS_DIR="$PWD/3rd_party/libs"
export LD_LIBRARY_PATH="$PWD/_lib:$PWD/3rd_party/libs:${LD_LIBRARY_PATH:-}"

pkill -f 'mortred-supervisor.out' 2>/dev/null || true
nohup ./_bin/mortred-supervisor.out > /tmp/mortred-supervisor.log 2>&1 &
sleep 3
curl -fsS -o /dev/null -w 'supervisor_health=%{http_code}\n' http://127.0.0.1:8787/api/v1/health
```

## 4. Auth smoke + infer

```bash
ROUTE=/mortred_ai_server_v1/classification/mobilenetv2
curl -s -o /dev/null -w 'unauth=%{http_code}\n' -X POST "http://127.0.0.1:8080${ROUTE}"
# expect 401

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

Put JSON in a **file**; do not put large base64 on argv.

## 5. doctor caveats

- `security_warn --self-test` fails if tokens already exported → `env -u` the three tokens then re-run.
- densenet 404: live probe greps first `server_uri`, not the demo pack.

Evidence: `docs/evidence/sme-02-shortest-path-cpu-20260914.md`.
