# UI 重设计期间的 C++ / API 变更记录

本项目纪律：凡改动 C++ 或 API，必须落盘。本轮 UI 重设计（Lumen）只有
**一处**后端改动。

## 1. `GET /api/v1/version`（新增，公开）

- **动机**：控制台侧边栏显示 `supervisor · v0.1.0` 版本徽章——产品信任感
  细节；该端点需在未配置 token 时可访问（控制台启动横幅），故与
  `/api/v1/health` 同级公开。
- **实现**：`src/control/supervisor/supervisor_app.cpp`
  - `#include "common/mortred_version.h"`（CMake 生成，单一事实来源为根
    CMakeLists 的 `project(VERSION)`）
  - 路由：`path == "/api/v1/version" && method == "GET"` →
    `{"component":"supervisor","version":MORTRED_VERSION}`
  - 鉴权白名单：`is_api && path != "/api/v1/health" && path != "/api/v1/version"`
    才要求 Bearer。
- **响应示例**：

```json
{"component": "supervisor", "version": "0.1.0"}
```

- **文档同步**：`docs/deployment.md` 端口表（8787 行注明 health/version 公开）
  与「Verify」命令清单各加一行。
- **前端消费**：`src/control/supervisor/ui/app.js` 启动时
  `fetch('/api/v1/version')` → `#ver-pill`。

## 静态资源版本号（非 API，工程备注）

`index.html` 对 `style.css` / `app.js` 的引用带 `?v=lumen<N>` 查询参数，
每轮迭代递增。C++ 静态服务按路径白名单（`/`、`/index.html`、`/app.js`、
`/style.css`）放行——带查询参数的请求 path 不含 query，路由不受影响，
无需改 C++。
