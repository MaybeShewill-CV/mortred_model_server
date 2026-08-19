# 全量 -Werror 门禁 + Console /ready 探测实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 完成复评遗留清单 #4（-Werror 门禁扩展到全量构建）与 #5（Console 就绪探测改调 `/ready` + README 拼写清理）。

**架构：** #4：本地已实测全量 `-Wall -Wextra -Werror` 仅 2 处 `-Wcomment`（注释内 `conf/server/*.toml` 的 glob `/*`），改写措辞即清零；CI 侧以 Dockerfile `build` 阶段为载体，新增 `EXTRA_CMAKE_FLAGS` 构建参数与 `full-werror` 作业（push 触发 + GHA 层缓存），并补 `full-werror` CMake preset 供本地一键复现。#5：新增头文件级 `ready_probe.h`（POSIX socket 短超时 GET /ready，200 即就绪），`ServerManager::is_ready` 从 grep 日志改为调用探测（端口取自 `_ports`），配独立单测（起假 HTTP 监听验证红绿），README 三处拼写修复。

**技术栈：** C++17、POSIX sockets、GTest、Docker buildx + GHA cache、WSL 验证环境。

---

### 任务 4：全量 -Werror 门禁

- [x] **步骤 1（已完成）**：本地实测 `build/werror-full`（FULL+WERROR）仅 4 行 `-Wcomment` 错误、2 个源码点；改写 catalog.cpp/catalog.h 注释后 **0 错误、100% 构建**。
- [ ] **步骤 2**：Dockerfile build 阶段加 `ARG EXTRA_CMAKE_FLAGS=""` 并注入 cmake。
- [ ] **步骤 3**：`CMakePresets.json` 加 `full-werror` preset（本地复现门禁）。
- [ ] **步骤 4**：ci.yml 新增 `full-werror` 作业：buildx `--target build` + `--build-arg EXTRA_CMAKE_FLAGS="-DMORTRED_ENABLE_WERROR=ON"` + GHA cache；触发条件 `push || (workflow_dispatch && full-matrix)`；同步更新 dispatch 输入描述。
- [ ] **步骤 5**：验证：`cmake --build build/werror-full` 全绿；`actionlint` 或 YAML 解析自检；Commit。

### 任务 5：Console /ready 探测 + README 清理

- [ ] **步骤 1（红）**：新建 `src/apps/web_console/backend/ready_probe.h`（`inline bool endpoint_ready(int port, const char* path, int timeout_ms)`），新增 `test/ready_probe_unittest.cc`：起真 socket 监听线程返回 200/503/无响应三种剧本；先写测试（函数未实现/断言现状）→ 编译运行确认失败。
- [ ] **步骤 2（绿）**：实现 `ready_probe.h`（connect + SO_RCVTIMEO + "GET path HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"，响应前缀含 `200` 即 true；连接被拒/超时/非 200 均 false）。
- [ ] **步骤 3**：`server_manager.{h,cpp}`：`is_ready` 改为 `_ports` 查端口 + `endpoint_ready(port, "/ready", 1000)`；删除日志扫描；更新头注释；main.cpp L342 错误文案改为 `/ready` 语义。
- [ ] **步骤 4**：README.md 三处拼写（Morted→Mortred、bellow→below、Torturials→Tutorials）。
- [ ] **步骤 5**：验证：单测全绿（新测试 + 既有 21 项）、`build/werror-full` 含新文件全绿、`check_consistency.py` 通过；Commit。

## 自检

#4 覆盖：本地清零（已实测）+ Docker 载体 + CI 作业 + preset 复现 ✓；#5 覆盖：探测函数（红绿单测）+ is_ready 替换 + 文案 + README 三处 ✓；无占位符 ✓。
