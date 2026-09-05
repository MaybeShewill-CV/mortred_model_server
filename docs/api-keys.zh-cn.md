# API Key 管理指南

| [English](api-keys.md) | [中文](api-keys.zh-cn.md) |
|---|---|

本指南涵盖 API Key 的完整生命周期：**服务端管理员**如何生成、配置和管理 key；**客户端用户**如何获取和使用 key 访问推理 API。

## 架构概览

```
┌──────────────┐     API Key (Bearer)      ┌──────────────┐
│  客户端用户   │ ─────────────────────────→ │    网关       │
│  (租户)      │   Authorization: Bearer... │   (:8080)    │
└──────────────┘                            └──────┬───────┘
                                            SHA-256 哈希查找
                                            作用域 + 限流检查
                                                   │
                                            ┌──────▼───────┐
                                            │  模型服务器    │
                                            │  (仅环回地址)  │
                                            └──────────────┘

┌──────────────────┐    改 toml 后重启网关    ┌──────────────┐
│ 服务端管理员      │  POST .../servers/...  │   监督器      │
│ (运维人员)       │  /restart              │   (:8787)    │
└──────────────────┘                        └──────────────┘
```

开发默认用 `mortredctl init-trust`（环境变量 token）。`conf/api_keys.toml` 是网关进程上的可选多租户鉴权。监督器 **没有** key 热加载：改文件后重启 gateway 子进程。`scope` 只决定该 Bearer **能不能推理**，不能管 :8787。

## 客户端用户指南

### 如何获取 API Key

1. **联系服务端管理员** — API Key 由 Mortred 部署的运维人员发放。你将收到：
   - API Key 字符串（如 `4f8a7b2c9d0e...`，64 位十六进制字符串）
   - 网关地址（如 `https://inference.example.com:8080`）
   - 你被分配的作用域（通常是 `inference`）和限流配额

2. **安全存储 Key**：
   ```bash
   # 保存到受限权限的文件
   echo "your-api-key-here" > ~/.mortred-api-key
   chmod 600 ~/.mortred-api-key

   # 或设置环境变量
   export MORTRED_API_KEY="your-api-key-here"
   ```

3. **Key 仅显示一次** — 服务端只存储 SHA-256 哈希，如果丢失 Key，必须申请新的。

### 如何使用 API Key

每个发往网关的请求必须携带 `Authorization: Bearer <key>` 头：

#### 同步推理（实时）

```bash
curl -X POST http://localhost:8080/mortred_ai_server_v1/obj_detection/yolov8 \
  -H "Authorization: Bearer $MORTRED_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"img_data": "'"$(base64 -w0 image.jpg)"'", "req_id": "my-request-1"}'
```

#### 异步任务（长耗时：扩散模型、SAM）

```bash
# 1. 提交
curl -X POST http://localhost:8080/v1/models/DDPM/jobs \
  -H "Authorization: Bearer $MORTRED_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"img_data": "...", "timestep": 100}'
# 返回: {"job_id": "job_xxx", "state": "pending"}

# 2. 轮询
curl http://localhost:8080/v1/models/DDPM/jobs/job_xxx \
  -H "Authorization: Bearer $MORTRED_API_KEY"

# 3. 取结果
curl http://localhost:8080/v1/models/DDPM/jobs/job_xxx/result \
  -H "Authorization: Bearer $MORTRED_API_KEY"
```

#### Python 示例

```python
import requests
import base64

API_KEY = "your-api-key-here"
GATEWAY = "http://localhost:8080"

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# 编码图片
with open("image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

# 推理
resp = requests.post(f"{GATEWAY}/mortred_ai_server_v1/obj_detection/yolov8",
                     headers=headers,
                     json={"img_data": img_b64, "req_id": "demo"})
print(resp.json())
```

### 响应头说明

| 响应头 | 说明 |
|---|---|
| `X-Mortred-Key` | 你的 key 名称（如 `tenant-a`）——确认使用了哪个 key |
| `X-Request-ID` | 回显的请求 ID，用于追踪 |
| `Retry-After` | 429 响应时出现——等待指定秒数后重试 |

### 错误响应

| 状态码 | 含义 | 处理方式 |
|---|---|---|
| 401 | Key 无效或缺失 | 检查 key 是否正确且已启用 |
| 429 | 超出限流配额 | 等待 `Retry-After` 秒后重试 |
| 404 | 未知的模型路径 | 与管理员确认模型 URI |
| 503 | 模型服务器未运行 | 联系管理员 |

### Key 轮换（Key 过期时）

1. 向管理员申请新 key
2. 更新环境变量：`export MORTRED_API_KEY="new-key"`
3. 旧 key 在管理员删除后停止工作
4. 零停机——过渡期间新旧 key 同时有效

---

## 服务端管理员指南

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

### 为客户端生成 API Key

当新客户端（租户）需要访问时：

```bash
# 第 1 步：生成随机 key（这个要交给客户端）
openssl rand -hex 32
# 示例输出: 3a7f9b2e8c4d1f6a0b5c3d8e2f7a4b9c6d1e0f3a5b8c2d7e4f1a6b3c8d0e5f

# 第 2 步：计算 SHA-256 哈希（这个写入配置文件）
echo -n "3a7f9b2e8c4d1f6a0b5c3d8e2f7a4b9c6d1e0f3a5b8c2d7e4f1a6b3c8d0e5f" | sha256sum
# 示例输出: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

# 第 3 步：将哈希写入 conf/api_keys.toml
```

```toml
# conf/api_keys.toml — 添加新客户端
[keys.new-client]
hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
scope = "inference"
rate_limit_qps = 100
enabled = true
```

```bash
# 第 4 步：重启 gateway 子进程以加载 conf/api_keys.toml
curl -X POST -H "Authorization: Bearer $MORTRED_API_TOKEN" \
  http://localhost:8787/api/v1/servers/__gateway/restart

# 第 5 步：将 key 字符串交给客户端（不是哈希！）
# 客户端使用:  Authorization: Bearer 3a7f9b2e8c4d...
# 你存储的是:   hash = "e3b0c44298fc..."
```

### 管理 Key

监督器 **不** 暴露 key 列表或用量计数。计数器在网关进程内，重启后清零。

#### 禁用 key（不删除）
```toml
# conf/api_keys.toml
[keys.suspended-client]
hash = "..."
enabled = false   # 下一次请求立即被拒
```
然后重启 gateway 子进程。

#### 重新启用 key
```toml
[keys.suspended-client]
hash = "..."
enabled = true
```
重启 gateway 子进程。

#### 删除 key
从 `conf/api_keys.toml` 中删除整个 `[keys.name]` 段，然后重启 gateway 子进程。

#### 修改 key 的限流配额
```toml
[keys.tenant-a]
hash = "..."
rate_limit_qps = 200   # 原来是 100
```
重启 gateway 子进程。对新请求立即生效。

### Key 轮换流程

文件里同时保留新旧 `[keys.*]`，重启 gateway，切换客户端，再删旧条目并再重启一次。重启期间网关会短暂不可用；一次重启时文件里同时有两个哈希，可避免客户端空窗。

### 配置文件安全

```bash
# 限制文件权限（仅服务用户可读）
sudo chown mortred:mortred conf/api_keys.toml
sudo chmod 600 conf/api_keys.toml

# 绝不将生产环境的 key 文件提交到版本控制
echo "conf/api_keys.toml" >> .gitignore
```

### 监控 Key 用量

每 key 计数在网关进程内，**:8787 不导出**。用网关访问日志（`X-Mortred-Key`）或 Prometheus HTTP 指标。网关子进程重启后计数清零。

### 故障排查

| 问题 | 原因 | 解决方式 |
|---|---|---|
| 所有请求 401 | 无 token、空 `api_keys.toml`、或 Bearer 错误 | `mortredctl init-trust` 或写入 `[keys.*]` 哈希；空/仅注释的 key 文件不算鉴权 |
| 新 key 不生效 | 网关仍在跑旧文件 | 重启 gateway 子进程 |
| Key 可用但返回 429 | 超出限流配额 | 增大 `rate_limit_qps` 并重启网关 |
| 客户端丢失 key | 只存储了哈希 | 生成新 key，禁用旧 key |
| 网关日志 "failed to parse" | TOML 语法错误 | 修好文件；没有静态 token 时网关拒绝启动 |
| 网关日志 "empty key file is not auth" | 复制了示例但没写哈希 | 补 key 或使用 init-trust token |

---

## 配置参考

### conf/api_keys.toml 格式

```bash
# 1. 生成随机 key
openssl rand -hex 32

# 2. 计算配置文件需要的哈希
echo -n "4f8a7b2c..." | sha256sum

# 3. 写入 conf/api_keys.toml
```

### 完整示例

```toml
# conf/api_keys.toml

# 实时推理客户端，中等速率
[keys.mobile-app]
hash = "a1b2c3d4e5f6..."
scope = "inference"
rate_limit_qps = 100
enabled = true

# 批处理客户端，高速率
[keys.batch-processor]
hash = "b2c3d4e5f6a7..."
scope = "inference"
rate_limit_qps = 500
enabled = true

# 运维 key（同一推理路径；不能解锁 :8787）
[keys.ops-team]
hash = "c3d4e5f6a7b8..."
scope = "all"
rate_limit_qps = 0
enabled = true

# 暂时停用的客户端
[keys.trial-expired]
hash = "d4e5f6a7b8c9..."
scope = "inference"
rate_limit_qps = 10
enabled = false
```

### 鉴权优先级

网关按顺序检查：
1. API Key（多 key）：哈希 Bearer token → 查 conf/api_keys.toml → 作用域检查 → 限流检查
2. 静态 token（旧版兼容）：与 MORTRED_GATEWAY_AUTH_TOKEN 比较

任一通过即授权。响应头 `X-Mortred-Key` 标识使用了哪个 key。

监督器没有 `/api/v1/keys` 接口。改完 `conf/api_keys.toml` 后重启 gateway 子进程：

```bash
curl -X POST -H "Authorization: Bearer $MORTRED_API_TOKEN" \
  http://localhost:8787/api/v1/servers/__gateway/restart
```

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

## Key 轮换

1. 生成新 key：`openssl rand -hex 32`
2. 将新哈希与旧 key 一并写入 conf/api_keys.toml
3. 重启 gateway 子进程
4. 把新 key 交给客户端
5. 删除旧 key 并再次重启 gateway 子进程

## 安全说明

- Key 绝不以明文存储——只有 SHA-256 哈希
- 配置文件应设为 chmod 600 并归服务用户所有
- 被禁用的 key 立即拒绝
- 限流为每 key 固定窗口（1 秒）
- 所有鉴权在网关层完成；模型服务器仅绑定环回地址
- fail-closed 只保证非环回监听必须配鉴权；不终结 TLS，也不隐藏网关 `/metrics`

## 并发与热加载

`ApiKeyManager::authenticate()` 返回 `shared_ptr<const ApiKey>`：调用方在
读取 key（name/scope/计数器）期间持有所有权，因此并发的 `reload()` 整体
替换 key 集合也不会使结果悬垂。调用方不得把裸指针保留到 shared_ptr 生命
期之外。`ApiKey` 上的运行时计数与限流状态是 `mutable` 的内部同步状态——
const key 仍会计数与限流，但其身份/配置永不变化。

该契约由 `test/api_key_manager_unittest.cc` 强制执行：压力测试让
authenticate() 与持续 reload 循环并发，并携带 `sanitizer` ctest label（CI
的 TSAN 门禁）。同一测试对旧的裸指针实现运行会在 ASan 下以
heap-use-after-free 崩溃。
