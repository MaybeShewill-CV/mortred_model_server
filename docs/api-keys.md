# API Key Management Guide

| [English](api-keys.md) | [中文](api-keys.zh-cn.md) |
|---|---|

This guide covers the complete lifecycle of API keys: how a **service administrator** generates, configures, and manages keys; and how a **client user** obtains and uses a key to access the inference API.

## Overview

```
┌──────────────┐     API Key (Bearer)      ┌──────────────┐
│  Client User │ ─────────────────────────→ │   Gateway    │
│  (tenant)    │   Authorization: Bearer... │   (:8080)    │
└──────────────┘                            └──────┬───────┘
                                            SHA-256 hash lookup
                                            scope + rate limit check
                                                   │
                                            ┌──────▼───────┐
                                            │ Model Server │
                                            │  (loopback)  │
                                            └──────────────┘

┌──────────────────┐    manage keys         ┌──────────────┐
│ Service Admin    │ ─────────────────────→ │  Supervisor  │
│ (operator)       │  /api/v1/keys{,/reload}│   (:8787)    │
└──────────────────┘                        └──────────────┘
```

## For Client Users

### How to Get an API Key

1. **Contact the service administrator** — API keys are issued by the operator of the Mortred deployment. You will receive:
   - Your API key string (e.g., `4f8a7b2c9d0e...`, a 64-character hex string)
   - The gateway address (e.g., `https://inference.example.com:8080`)
   - Your assigned scope (typically `inference`) and rate limit

2. **Store the key securely**:
   ```bash
   # Save to a file with restricted permissions
   echo "your-api-key-here" > ~/.mortred-api-key
   chmod 600 ~/.mortred-api-key

   # Or set as an environment variable
   export MORTRED_API_KEY="your-api-key-here"
   ```

3. **The key is shown to you only once** — the server stores only its SHA-256 hash, so if you lose the key, you must request a new one.

### How to Use Your API Key

Every request to the gateway must include the `Authorization: Bearer <key>` header:

#### Sync Inference (real-time)

```bash
curl -X POST http://localhost:8080/mortred_ai_server_v1/obj_detection/yolov8 \
  -H "Authorization: Bearer $MORTRED_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"img_data": "'"$(base64 -w0 image.jpg)"'", "req_id": "my-request-1"}'
```

#### Async Job (long-running: diffusion, SAM)

```bash
# 1. Submit
curl -X POST http://localhost:8080/mortred_ai_server_v1/diffusion/ddpm/jobs \
  -H "Authorization: Bearer $MORTRED_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"img_data": "...", "timestep": 100}'
# Returns: {"job_id": "job_xxx", "state": "pending"}

# 2. Poll
curl http://localhost:8080/mortred_ai_server_v1/diffusion/ddpm/jobs/job_xxx \
  -H "Authorization: Bearer $MORTRED_API_KEY"

# 3. Get result
curl http://localhost:8080/mortred_ai_server_v1/diffusion/ddpm/jobs/job_xxx/result \
  -H "Authorization: Bearer $MORTRED_API_KEY"
```

#### Python Example

```python
import requests
import base64

API_KEY = "your-api-key-here"
GATEWAY = "http://localhost:8080"

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# Encode image
with open("image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

# Inference
resp = requests.post(f"{GATEWAY}/mortred_ai_server_v1/obj_detection/yolov8",
                     headers=headers,
                     json={"img_data": img_b64, "req_id": "demo"})
print(resp.json())
```

### Response Headers

| Header | Description |
|---|---|
| `X-Mortred-Key` | Your key name (e.g., `tenant-a`) — confirms which key was used |
| `X-Request-ID` | Echoed request ID for tracing |
| `Retry-After` | Present on 429 responses — wait this many seconds before retrying |

### Error Responses

| Status | Meaning | What to do |
|---|---|---|
| 401 | Invalid or missing key | Check that your key is correct and enabled |
| 429 | Rate limit exceeded | Wait `Retry-After` seconds, then retry |
| 404 | Unknown model path | Check the model URI with the administrator |
| 503 | Model server not running | Contact the administrator |

### Key Rotation (when your key expires)

1. Request a new key from the administrator
2. Update your environment: `export MORTRED_API_KEY="new-key"`
3. The old key stops working after the administrator removes it
4. No downtime — both keys work simultaneously during the transition

---

## For Service Administrators

## Configuration

API keys are defined in `conf/api_keys.toml`:

```toml
[keys.tenant-a]
# SHA-256 hash of the API key string (never store the plaintext key)
# Generate: echo -n "your-secret-key" | sha256sum
hash = "a1b2c3d4..."
scope = "inference"       # inference | admin | all
rate_limit_qps = 100      # 0 = unlimited
enabled = true
```

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `hash` | string | required | SHA-256 hex of the key |
| `scope` | string | "inference" | "inference", "admin", or "all" |
| `rate_limit_qps` | int | 0 | Per-key requests per second (0 = unlimited) |
| `enabled` | bool | true | Disable a key without deleting it |

### Generating an API Key for a Client

When a new client (tenant) needs access:

```bash
# Step 1: Generate a random key (this is what you give to the client)
openssl rand -hex 32
# Example output: 3a7f9b2e8c4d1f6a0b5c3d8e2f7a4b9c6d1e0f3a5b8c2d7e4f1a6b3c8d0e5f

# Step 2: Compute the SHA-256 hash (this is what goes in the config)
echo -n "3a7f9b2e8c4d1f6a0b5c3d8e2f7a4b9c6d1e0f3a5b8c2d7e4f1a6b3c8d0e5f" | sha256sum
# Example output: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

# Step 3: Record the hash in conf/api_keys.toml
```

```toml
# conf/api_keys.toml — add the new client
[keys.new-client]
hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
scope = "inference"
rate_limit_qps = 100
enabled = true
```

```bash
# Step 4: Hot-reload (no gateway restart needed)
curl -X POST -H "Authorization: Bearer $MORTRED_API_TOKEN" \
  http://localhost:8787/api/v1/keys/reload

# Step 5: Give the key string to the client (NOT the hash)
# The client uses: Authorization: Bearer 3a7f9b2e8c4d...
# You store:       hash = "e3b0c44298fc..."
```

### Managing Keys

#### List all keys
```bash
curl -H "Authorization: Bearer $MORTRED_API_TOKEN" \
  http://localhost:8787/api/v1/keys
```

Response:
```json
{
  "keys": [
    {"name": "tenant-a", "scope": "inference", "enabled": true,
     "total_requests": 15234, "total_rejected": 12},
    {"name": "admin", "scope": "all", "enabled": true,
     "total_requests": 45, "total_rejected": 0}
  ]
}
```

#### Disable a key (without deleting)
```toml
# conf/api_keys.toml
[keys.suspended-client]
hash = "..."
enabled = false   # immediately rejected on next request
```
Then reload: `POST /api/v1/keys/reload`

#### Re-enable a key
```toml
[keys.suspended-client]
hash = "..."
enabled = true
```
Reload.

#### Delete a key
Remove the entire `[keys.name]` section from `conf/api_keys.toml`, then reload.

#### Change a key's rate limit
```toml
[keys.tenant-a]
hash = "..."
rate_limit_qps = 200   # was 100
```
Reload. Takes effect immediately for new requests.

### Key Rotation Procedure (zero downtime)

```
Before:  [keys.tenant-a] hash=old_hash

Step 1:  [keys.tenant-a]  hash=old_hash          ← keep old
         [keys.tenant-a-v2] hash=new_hash        ← add new
         → reload → both keys work

Step 2:  Client switches to new key (old still works as fallback)

Step 3:  [keys.tenant-a-v2] hash=new_hash        ← only new
         (remove [keys.tenant-a])
         → reload → old key rejected
```

### Configuration File Security

```bash
# Restrict file permissions (only the service user can read)
sudo chown mortred:mortred conf/api_keys.toml
sudo chmod 600 conf/api_keys.toml

# Never commit the production key file to version control
# (add to .gitignore if deploying from git)
echo "conf/api_keys.toml" >> .gitignore
```

### Monitoring Key Usage

```bash
# Check per-key request counts
curl -H "Authorization: Bearer $MORTRED_API_TOKEN" \
  http://localhost:8787/api/v1/keys | jq '.keys[] | {name, total_requests, total_rejected}'

# Set up a Prometheus alert for high rejection rates
# (already in deploy/alert-rules.yml → MortredOverloadRejections)
```

### Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| All requests return 401 | api_keys.toml not found or empty | Check file path and content |
| New key doesn't work | Not reloaded after config change | `POST /api/v1/keys/reload` |
| Key works but returns 429 | Rate limit reached | Increase `rate_limit_qps` in config |
| Client lost their key | Only hash is stored | Generate a new key, disable the old one |
| Gateway logs "failed to parse" | TOML syntax error | Check `hash = "..."` quoting |

---

## Configuration Reference

### conf/api_keys.toml Format

```bash
# 1. Generate a random key
openssl rand -hex 32

# 2. Compute the hash for the config file
echo -n "4f8a7b2c..." | sha256sum

# 3. Add to conf/api_keys.toml
```

### Complete Example

```toml
# conf/api_keys.toml

# Real-time inference client with moderate rate
[keys.mobile-app]
hash = "a1b2c3d4e5f6..."
scope = "inference"
rate_limit_qps = 100
enabled = true

# Batch processing client with high rate
[keys.batch-processor]
hash = "b2c3d4e5f6a7..."
scope = "inference"
rate_limit_qps = 500
enabled = true

# Operations admin (can access management endpoints)
[keys.ops-team]
hash = "c3d4e5f6a7b8..."
scope = "all"
rate_limit_qps = 0
enabled = true

# Temporarily suspended client
[keys.trial-expired]
hash = "d4e5f6a7b8c9..."
scope = "inference"
rate_limit_qps = 10
enabled = false
```

### Authentication Priority

The gateway checks in order:
1. API Key (multi-key): hash the Bearer token → lookup in conf/api_keys.toml → check scope → check rate limit
2. Static token (legacy): compare with MORTRED_GATEWAY_AUTH_TOKEN

If either succeeds, the request is authorized. The `X-Mortred-Key` response header identifies which key was used.

### Management API Summary

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/keys` | GET | List all keys (name, scope, enabled, usage) |
| `/api/v1/keys/reload` | POST | Hot-reload after config change |

Both require the supervisor's `MORTRED_API_TOKEN` (admin credential).

```bash
curl -X POST http://localhost:8080/mortred_ai_server_v1/obj_detection/yolov8 \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"img_data": "base64..."}'
```

## Authentication Priority

The gateway checks in order:
1. API Key (multi-key): hash the Bearer token, look up in conf/api_keys.toml, check scope and rate limit
2. Static token (legacy): compare with MORTRED_GATEWAY_AUTH_TOKEN

If either succeeds, the request is authorized. The `X-Mortred-Key` response header identifies which key was used.

## Management API

Via the supervisor (:8787, requires MORTRED_API_TOKEN):

```bash
# List all keys (name, scope, enabled, usage - never the hash)
curl -H "Authorization: Bearer $ADMIN_TOKEN" http://localhost:8787/api/v1/keys

# Hot-reload after editing conf/api_keys.toml
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" http://localhost:8787/api/v1/keys/reload
```

## Key Rotation

1. Generate a new key: `openssl rand -hex 32`
2. Add the new hash alongside the old key in conf/api_keys.toml
3. Hot-reload: `POST /api/v1/keys/reload`
4. Distribute the new key to the client
5. Remove the old key and reload again

## Security Notes

- Keys are never stored in plaintext - only SHA-256 hashes
- The key file should be chmod 600 and owned by the service user
- Disabled keys are rejected immediately
- Rate limiting is per-key fixed-window (1 second)
- All authentication happens at the gateway; model servers are loopback-only behind the internal token

## Concurrency and Hot Reload

`ApiKeyManager::authenticate()` returns a `shared_ptr<const ApiKey>`: the caller
owns the key for as long as it reads it (name/scope/counters), so a concurrent
`reload()` swapping the whole key set can never dangle the result. Callers must
not keep the raw pointer beyond the shared_ptr's lifetime. Runtime counters and
rate-limiter state on `ApiKey` are `mutable` internal synchronization state - a
const key still counts and rate-limits, but its identity/config never changes.

This contract is enforced by `test/api_key_manager_unittest.cc`: a stress test
drives authenticate() against a continuous reload loop and carries the
`sanitizer` ctest label (TSAN gate in CI). The same test against the previous
raw-pointer implementation crashes with a heap-use-after-free under ASan.
