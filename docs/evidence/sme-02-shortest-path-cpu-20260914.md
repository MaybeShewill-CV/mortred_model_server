# SME-02 evidence — cpu shortest success path (source tree)

- **Date:** 2026-09-14
- **Host:** DESKTOP-B8MJFUS (WSL2), `/mnt/g/Codex/mortred_model_server`
- **Git:** `224a460` (docs land on follow-up commit)
- **Profile:** `cpu` (`MORTRED_PROFILE=cpu`, pack `conf/packs/demo.toml` → MOBILENETV2)

## Path run

1. `cmake --preset full-cpu` + `cmake --build build/full-cpu` → control-plane + `mortred-model-server.out`
2. `./scripts/mortredctl_init-trust.sh --force` → `conf/local/trust.env`
3. Weights: `weights/classification/mobilenetv2/mobilenetv2_ilsvrc2012.mnn` present
4. Supervisor with `MORTRED_PACK=conf/packs/demo.toml` + `LD_LIBRARY_PATH=_lib:3rd_party/libs` → `:8787`, health **200**
5. Unauthenticated `POST /mortred_ai_server_v1/classification/mobilenetv2` → **401**
6. Authenticated infer, body file from `demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG` as JSON `images` base64 → body **`status:0` OK**, `model.name=MOBILENETV2` (e.g. top category Irish wolfhound)

## doctor notes (not blocking this path)

| Check | Result | Interpretation |
|---|---|---|
| `security_warn.sh --self-test` | FAIL if tokens exported | Env pollution; passes under `env -u MORTRED_*_TOKEN` then `--self-test` |
| densenet unauth expect 401 got **404** | FAIL | `verify_deployment.sh` uses first `server_uri` in `conf/server`, not demo pack |

## GPU

Not run (`install_deps.sh --all` gpu previously Cutlass-blocked).

## Acceptance

**Met** for cpu source-tree shortest path.
