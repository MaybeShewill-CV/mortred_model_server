# 路线 B：moodycamel v1.0.4 阻塞队列改造实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 把 `3rd_party` 里的 moodycamel 从精简变体整体升级为上游 v1.0.4 三件套（`concurrentqueue.h` + `blockingconcurrentqueue.h` + `lightweightsemaphore.h`），用 `BlockingConcurrentQueue` 消除两处忙等（`base_server_impl.h` 的 worker 池、`sam_amg_decoder.cpp` 的解码线程队列），并给 worker 等待加上可选的有界超时背压。

**架构：** 保持现有"worker 池 + Workflow go task"的服务模型不变，只把 worker 队列从无锁 `ConcurrentQueue` 换成带信号量的 `BlockingConcurrentQueue`：取 worker 由 `while (!try_dequeue(worker)) {}` 忙等改为 `wait_dequeue` 阻塞等待（`_m_model_run_timeout > 0` 时用 `wait_dequeue_timed` 做有界等待），还 worker 直接 `enqueue`。26 个具体服务的 init 只调用 `_m_working_queue.enqueue(std::move(worker))`，接口兼容、零改动。

**技术栈：** C++17、moodycamel v1.0.4（BSD/Boost + zlib 许可）、GTest（契约测试）、CMake + Workflow（Linux 全量验证）、mingw g++（Windows 本地红-绿验证）。

---

## 文件结构

| 操作 | 文件 | 职责 |
|---|---|---|
| 替换 | `3rd_party/include/stl_container/concurrentqueue.h` | 上游 v1.0.4 无锁队列（现为精简变体） |
| 新增 | `3rd_party/include/stl_container/blockingconcurrentqueue.h` | 阻塞封装，提供 `wait_dequeue` / `wait_dequeue_timed` |
| 新增 | `3rd_party/include/stl_container/lightweightsemaphore.h` | `blockingconcurrentqueue.h` 的信号量依赖 |
| 修改 | `src/server/base_server_impl.h` | include、成员类型、`do_work` 两处忙等 |
| 修改 | `src/models/segment_anything/sam_automask_generator/sam_amg_decoder.cpp` | include、成员类型、忙等 |
| 新增 | `test/blocking_worker_queue_unittest.cc` | 阻塞队列契约测试 |
| 修改 | `test/CMakeLists.txt` | `TEST_LIST` 注册新测试 |

背景事实（已核实）：moodycamel 上游 `concurrentqueue.h` 本身不含 `BlockingConcurrentQueue` 实现，只有前置声明；完整实现在同仓库 `blockingconcurrentqueue.h`，它通过 `#include "concurrentqueue.h"` 和 `#include "lightweightsemaphore.h"` 拉依赖，三个文件必须同目录。仓库现有精简版与上游任何 tag 在忽略空白后仍差约 850-900 行，故采用"整体替换为 v1.0.4"。

---

### 任务 1：vendor 上游 v1.0.4 三件套 + 阻塞队列契约测试

**文件：**
- 创建：`test/blocking_worker_queue_unittest.cc`
- 修改：`test/CMakeLists.txt`
- 创建：`3rd_party/include/stl_container/blockingconcurrentqueue.h`
- 创建：`3rd_party/include/stl_container/lightweightsemaphore.h`
- 修改：`3rd_party/include/stl_container/concurrentqueue.h`（整体替换为 v1.0.4 内容）

> 此任务先写测试（红灯：头文件不存在），再同步头文件（绿灯），保证契约测试真的覆盖了最终 vendor 的文件。

- [ ] **步骤 1：编写失败的测试**

创建 `test/blocking_worker_queue_unittest.cc`：

```cpp
/************************************************
 * Author: Codex
 * File: blocking_worker_queue_unittest.cc
 * Date: 2026-08-11
 ************************************************/

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include "stl_container/blockingconcurrentqueue.h"

using Queue = moodycamel::BlockingConcurrentQueue<std::unique_ptr<int> >;

TEST(blocking_worker_queue, enqueue_then_wait_dequeue) {
    Queue q;
    q.enqueue(std::unique_ptr<int>(new int(42)));
    std::unique_ptr<int> item;
    q.wait_dequeue(item);
    ASSERT_NE(item, nullptr);
    EXPECT_EQ(*item, 42);
}

TEST(blocking_worker_queue, wait_blocks_until_wakeup) {
    Queue q;
    std::atomic<bool> done{false};
    std::thread waiter([&] {
        std::unique_ptr<int> item;
        q.wait_dequeue(item);
        done = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_FALSE(done.load()) << "空队列时 wait_dequeue 不应立即返回";
    q.enqueue(std::unique_ptr<int>(new int(7)));
    waiter.join();
    EXPECT_TRUE(done.load()) << "enqueue 后等待者应被唤醒";
}

TEST(blocking_worker_queue, timed_wait_timeout) {
    Queue q;
    std::unique_ptr<int> item;
    bool ok = q.wait_dequeue_timed(item, std::chrono::milliseconds(50));
    EXPECT_FALSE(ok);
}

TEST(blocking_worker_queue, fifo_order) {
    Queue q;
    for (int i = 1; i <= 3; ++i) {
        q.enqueue(std::unique_ptr<int>(new int(i)));
    }
    for (int i = 1; i <= 3; ++i) {
        std::unique_ptr<int> item;
        q.wait_dequeue(item);
        ASSERT_NE(item, nullptr);
        EXPECT_EQ(*item, i);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

同时把 `json_request_parser_unittest` 换成新增项后的列表写进 `test/CMakeLists.txt`：

```cmake
set(TEST_LIST
    base64_unittest
    md5_unittest
    file_path_util_unittest
    json_request_parser_unittest
    blocking_worker_queue_unittest
)
```

- [ ] **步骤 2：运行测试验证失败**

本机（Windows，mingw g++，无 GTest）用临时垫片执行。先创建 `$env:TEMP\gtest_shim\gtest\gtest.h`（垫片仅用于本地红-绿验证，不进仓库）：

```cpp
// 临时 GTest 兼容垫片（仅本地验证用）
#pragma once
#include <cstdio>
static int g_test_ran = 0;
static int g_test_failures = 0;
#define TEST(suite, name)                                                    \
    static void suite##_##name();                                            \
    struct suite##_##name##_registrar {                                      \
        suite##_##name##_registrar() { ++g_test_ran; suite##_##name(); }     \
    } suite##_##name##_registrar_instance;                                   \
    static void suite##_##name()
#define EXPECT_TRUE(cond)                                                    \
    do {                                                                     \
        ++g_test_ran;                                                        \
        if (!(cond)) {                                                       \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
            ++g_test_failures;                                               \
        }                                                                    \
    } while (0)
#define EXPECT_FALSE(cond) EXPECT_TRUE(!(cond))
#define EXPECT_EQ(a, b)                                                      \
    do {                                                                     \
        auto _a = (a);                                                       \
        auto _b = (b);                                                       \
        ++g_test_ran;                                                        \
        if (!(_a == _b)) {                                                   \
            std::printf("FAIL %s:%d: %s == %s\n", __FILE__, __LINE__,        \
                        #a, #b);                                             \
            ++g_test_failures;                                               \
        }                                                                    \
    } while (0)
#define ASSERT_NE(a, b) EXPECT_TRUE(!((a) == (b)))
#define ASSERT_TRUE(cond) EXPECT_TRUE(cond)
#define ASSERT_EQ(a, b) EXPECT_EQ((a), (b))
namespace testing {
inline void InitGoogleTest(int*, char**) {}
inline int RUN_ALL_TESTS() {
    std::printf("assertions ran: %d, failures: %d\n", g_test_ran,
                g_test_failures);
    return g_test_failures == 0 ? 0 : 1;
}
} // namespace testing
#define RUN_ALL_TESTS() ::testing::RUN_ALL_TESTS()
```

编译并运行（此时 `blockingconcurrentqueue.h` 尚不存在）：

```powershell
& 'E:\compile_tools\mingw64\bin\g++.exe' -std=c++17 -pthread `
  -I $env:TEMP\gtest_shim -I 3rd_party\include `
  test\blocking_worker_queue_unittest.cc -o $env:TEMP\bwq_test.exe
```

预期：编译失败，报 `stl_container/blockingconcurrentqueue.h: No such file or directory`——失败原因是功能缺失（正确红灯），不是拼写错误。

- [ ] **步骤 3：同步三个头文件**

来源（全部锁定 `v1.0.4` tag）：

| 文件 | 下载地址 | SHA-256 |
|---|---|---|
| `concurrentqueue.h` | `https://raw.githubusercontent.com/cameron314/concurrentqueue/v1.0.4/concurrentqueue.h` | `524BC2DE581ECC95E632EE6AAC2676D6A1029A692A9C1B4A32E4A80608ABAD5E` |
| `blockingconcurrentqueue.h` | `https://raw.githubusercontent.com/cameron314/concurrentqueue/v1.0.4/blockingconcurrentqueue.h` | `27CE49DFBFE01F0DB9E505E55772418AFB90AC2466CAA99BDE8B3FE63EA2F936` |
| `lightweightsemaphore.h` | `https://raw.githubusercontent.com/cameron314/concurrentqueue/v1.0.4/lightweightsemaphore.h` | `0B78024F079A6B43FD9373B9402BE3BC18291BF6B5BC391ACF35D1EF89D324AE` |

把三个文件下载到本地目录（例如 `$env:TEMP\moodycamel_v104`），校验哈希与上表一致后，原样复制进 `3rd_party/include/stl_container/`（保留每个文件的许可头，不做任何裁剪）：

```powershell
$src = "$env:TEMP\moodycamel_v104"
Copy-Item -LiteralPath "$src\concurrentqueue.h"         -Destination 3rd_party\include\stl_container\concurrentqueue.h -Force
Copy-Item -LiteralPath "$src\blockingconcurrentqueue.h" -Destination 3rd_party\include\stl_container\blockingconcurrentqueue.h -Force
Copy-Item -LiteralPath "$src\lightweightsemaphore.h"    -Destination 3rd_party\include\stl_container\lightweightsemaphore.h -Force
```

```bash
# Linux 等价命令
SRC=/tmp/moodycamel_v104
cp $SRC/concurrentqueue.h         3rd_party/include/stl_container/concurrentqueue.h
cp $SRC/blockingconcurrentqueue.h 3rd_party/include/stl_container/blockingconcurrentqueue.h
cp $SRC/lightweightsemaphore.h    3rd_party/include/stl_container/lightweightsemaphore.h
```

- [ ] **步骤 4：运行测试验证通过**

重新执行步骤 2 的编译与运行命令（Windows）或 Linux 等价命令：

```bash
g++ -std=c++17 -pthread -I3rd_party/include test/blocking_worker_queue_unittest.cc \
    -lgtest -lgtest_main -o /tmp/bwq_test && /tmp/bwq_test
```

预期：4 个用例全部通过（本机垫片输出 `assertions ran: N, failures: 0`）。

- [ ] **步骤 5：Commit**

```bash
git add 3rd_party/include/stl_container/concurrentqueue.h \
        3rd_party/include/stl_container/blockingconcurrentqueue.h \
        3rd_party/include/stl_container/lightweightsemaphore.h \
        test/blocking_worker_queue_unittest.cc test/CMakeLists.txt
git commit -m "chore(3rd_party): sync moodycamel to v1.0.4 with blocking queue"
```

---

### 任务 2：`base_server_impl.h` 忙等修复（含可选有界等待）

**文件：**
- 修改：`src/server/base_server_impl.h:13`（include 行）
- 修改：`src/server/base_server_impl.h:104`（成员类型）
- 修改：`src/server/base_server_impl.h:235` 与 `:260`（`do_work` 两处忙等）

- [ ] **步骤 1：修改 include 与成员类型**

include 行（第 13 行）改为：

```cpp
#include "stl_container/blockingconcurrentqueue.h"
```

在 include 区补充 `<chrono>`（当前文件没有显式包含它，`wait_dequeue_timed` 需要）：

```cpp
#include <chrono>
```

成员类型（第 104 行）改为：

```cpp
moodycamel::BlockingConcurrentQueue<WORKER> _m_working_queue;
```

- [ ] **步骤 2：修改 `do_work` 的取/还 worker**

`do_work` 中"取 worker"部分（原第 235 行附近）替换为：

```cpp
    WORKER worker;
    auto find_worker_start_ts = Timestamp::now();

    if (_m_model_run_timeout > 0) {
        // 有界等待：等 worker 也计入模型超时预算，形成背压
        if (!_m_working_queue.wait_dequeue_timed(
                worker, std::chrono::milliseconds(_m_model_run_timeout))) {
            ctx->model_run_status = StatusCode::MODEL_RUN_TIMEOUT;
            ctx->task_finished_ts = Timestamp::now().to_format_str();
            // 关键：提前退出也必须恰好计一次 release_ctx，否则 ctx 泄漏
            WFTaskFactory::count_by_name("release_ctx");
            return;
        }
    } else {
        // model_run_timeout <= 0 表示不设超时，用无界阻塞等待
        _m_working_queue.wait_dequeue(worker);
    }
    ctx->find_worker_time_consuming = (Timestamp::now() - find_worker_start_ts) * 1000;
```

"还 worker"部分（原第 260 行附近）替换为：

```cpp
    _m_working_queue.enqueue(std::move(worker));
```

其余逻辑（时间戳、`model_input` 构造、`worker->run`、日志里的 `size_approx()`）保持不变。

- [ ] **步骤 3：静态检查无忙等残留**

```bash
rg -n "while \(!.*try_dequeue|while \(!.*\.enqueue" src/server/base_server_impl.h
```

预期：无输出。

- [ ] **步骤 4：Linux 全量编译验证**

```bash
cd build && make -j10
```

预期：编译通过（含全部 server 目标；26 个具体服务 init 的 `enqueue()` 调用零改动）。

> Windows 本机无法编译此文件（依赖系统 glog 与 Linux 版 Workflow 动态库），所以编译与行为验证必须在 Linux 构建环境完成。

- [ ] **步骤 5：行为验证（并发压测观察 CPU）**

```bash
./_bin/mobilenetv2_classification_server.out \
  ../conf/server/classification/mobilenetv2/mobilenetv2_server_config.toml &
# 另开终端
python ../scripts/server/test_server.py --server mobilenetv2 --mode single
top -H -p <server_pid>
```

预期：并发请求超过 `worker_nums` 时，等待中的 compute 线程 CPU 占用接近 0%（修复前是 100% 空转）。可用 `build/tmp` 下的 `worker_pool_demo`（`make run`）做直观对照。

- [ ] **步骤 6：Commit**

```bash
git add src/server/base_server_impl.h
git commit -m "fix(server): replace busy-wait worker dequeue with blocking wait"
```

---

### 任务 3：`sam_amg_decoder.cpp` 忙等修复

**文件：**
- 修改：`src/models/segment_anything/sam_automask_generator/sam_amg_decoder.cpp:11`（include 行）
- 修改：`src/models/segment_anything/sam_automask_generator/sam_amg_decoder.cpp:129`（成员类型）
- 修改：`src/models/segment_anything/sam_automask_generator/sam_amg_decoder.cpp:662`（忙等）

- [ ] **步骤 1：修改 include 与成员类型**

include 行（第 11 行）：

```cpp
#include "stl_container/blockingconcurrentqueue.h"
```

成员类型（第 129 行）：

```cpp
moodycamel::BlockingConcurrentQueue<ThreadExecutor> _m_decoder_queue;
```

- [ ] **步骤 2：修改忙等**

第 662 行：

```cpp
_m_decoder_queue.wait_dequeue(decode_executor);
```

第 316/802 行的 `_m_decoder_queue.enqueue(executor)` 保持不变（接口兼容）。已核实该队列的消费者 `thread_decode_mask_proc` 总是把 executor 归还队列，不存在"取走不还"导致永久阻塞的路径。

- [ ] **步骤 3：静态检查**

```bash
rg -n "while \(!.*try_dequeue" src/models/segment_anything/sam_automask_generator/sam_amg_decoder.cpp
```

预期：无输出。

- [ ] **步骤 4：Linux 编译 + SAM benchmark 验证**

```bash
cd build && make -j10
./_bin/sam_benchmark.out ../conf/model/segment_anything/mobile_sam_config.toml
```

预期：编译通过；SAM benchmark 正常输出解码结果（覆盖 `thread_decode_mask_proc` 路径）。

- [ ] **步骤 5：Commit**

```bash
git add src/models/segment_anything/sam_automask_generator/sam_amg_decoder.cpp
git commit -m "fix(sam): replace busy-wait decoder dequeue with blocking wait"
```

---

### 任务 4：全量回归与收尾检查

**文件：** 无（纯检查任务；如发现问题则回到对应任务修复并各自 commit）

- [ ] **步骤 1：全仓忙等残留扫描**

```bash
rg -n "while \(!.*try_dequeue|while \(!.*\.enqueue" src
```

预期：无输出。

- [ ] **步骤 2：Linux 全量构建 + 冒烟**

```bash
cd build && make -j10
./_bin/mobilenetv2_classification_server.out ../conf/server/classification/mobilenetv2/mobilenetv2_server_config.toml &
./_bin/sam_benchmark.out ../conf/model/segment_anything/mobile_sam_config.toml
./_bin/yolov5_detection_server.out ../conf/server/object_detection/yolov5/yolov5_server_config.toml &
```

预期：全部目标编译通过；三个服务/工具正常启动与运行。

- [ ] **步骤 3：git 状态确认**

```bash
git status --short
```

预期：仅包含任务 1-3 提交的文件；`build/tmp`（demo）与 `$env:TEMP` 垫片不在提交范围（`build/` 已在 `.gitignore` 中）。若发现多余改动，清理后再结束。
