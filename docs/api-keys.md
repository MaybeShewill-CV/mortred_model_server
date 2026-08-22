# API Key Management

| [English](api-keys.md) | [中文](api-keys.zh-cn.md) |
|---|---|

Multi-tenant API key authentication for the inference gateway. Keys are stored as SHA-256 hashes with per-key scope, rate limit, and usage tracking.

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

## Generating a Key

```bash
# 1. Generate a random key
openssl rand -hex 32

# 2. Compute the hash for the config file
echo -n "4f8a7b2c..." | sha256sum

# 3. Add to conf/api_keys.toml
```

## Using a Key

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
