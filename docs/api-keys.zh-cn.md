# API Key 管理

| [English](api-keys.md) | [中文](api-keys.zh-cn.md) |
|---|---|

推理网关的多租户 API Key 鉴权。Key 以 SHA-256 哈希存储，支持按 key 的作用域、限流与用量追踪。

## 配置

API Key 定义在 `conf/api_keys.toml`：

```toml
[keys.tenant-a]
# API Key 字符串的 SHA-256 哈希（绝不存储明文）
hash = "a1b2c3d4..."
scope = "inference"       # inference | admin | all
rate_limit_qps = 100      # 0 = 不限
enabled = true
```

### 字段说明

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `hash` | string | 必填 | Key 的 SHA-256 十六进制 |
| `scope` | string | "inference" | "inference"（仅推理）、"admin"（管理）、"all"（两者） |
| `rate_limit_qps` | int | 0 | 每 key 每秒请求上限（0 = 不限） |
| `enabled` | bool | true | 禁用 key 而不删除 |

## 生成 Key

```bash
# 1. 生成随机 key
openssl rand -hex 32

# 2. 计算配置文件需要的哈希
echo -n "4f8a7b2c..." | sha256sum

# 3. 写入 conf/api_keys.toml
```

## 使用 Key

```bash
curl -X POST http://localhost:8080/mortred_ai_server_v1/obj_detection/yolov8 \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"img_data": "base64..."}'
```

## 鉴权优先级

网关按顺序检查：
1. API Key（多 key）：哈希 Bearer token → 查 conf/api_keys.toml → 作用域 → 限流
2. 静态 token（旧版兼容）：与 MORTRED_GATEWAY_AUTH_TOKEN 比较

任一通过即授权。响应头 `X-Mortred-Key` 标识使用了哪个 key。

## 管理 API

通过监督器（:8787，需要 MORTRED_API_TOKEN）：

```bash
# 列出全部 key
curl -H "Authorization: Bearer $ADMIN_TOKEN" http://localhost:8787/api/v1/keys

# 热加载
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" http://localhost:8787/api/v1/keys/reload
```

## Key 轮换

1. 生成新 key
2. 将新哈希加入 conf/api_keys.toml（与旧 key 并存）
3. 热加载
4. 分发新 key 给客户端
5. 删除旧 key 并再次热加载

## 安全说明

- Key 绝不以明文存储——只有 SHA-256 哈希
- 配置文件应设为 chmod 600 并归服务用户所有
- 被禁用的 key 立即拒绝
- 限流为每 key 固定窗口（1 秒）
- 所有鉴权在网关层完成；模型服务器仅绑定环回地址
