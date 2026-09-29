# 技术方案

- 问题: P1-3 已有 fail-closed 黑盒脚本未进默认 ctest
- 版本: v1
- 对应列表: `cowork/ledger/round-004/02-p0-p1-final.md`（P1-3）

## 边界

- 只把已存在的 `scripts/test_p0_1_engine_fail_closed.sh` 挂进默认 ctest 可发现、可执行的用例；不重做 P0-1 产品逻辑，不改 AC1–AC8 判定语义。
- 脚本现有退出码约定必须保持：内部 `skip`（含 AC8 无 GPU/`trtexec`/onnx）不计入 `FAILS`；`FAILS=0` 时 `exit 0`，有失败才 `exit 1`。不得改成「无 GPU 时非 0」或依赖 CTest `SKIP_RETURN_CODE` 改写该约定。
- 默认合入门仍是 `check` → `ctest --output-on-failure -LE valgrind`；新用例不得打 `valgrind` label，否则会被默认门排除。
- tests-only 与 FULL 都会处理 `test/CMakeLists.txt`；本用例是仓库根目录黑盒脚本，不依赖 models/server 链接，须在两种构建下均可注册。
- 不改 `src/`。不做训练/标注/非 Linux。不把 `--help` 当成安装或挂测成功。不扩大到 P1-1/P1-2 或其他列表项。

## 做法

### 推荐

在 `test/CMakeLists.txt` 于现有 gtest `foreach`/`add_test` 之外，无条件再注册一条 shell 黑盒 `add_test`：

- `NAME` 固定为 `p0_1_engine_fail_closed`（可被 `ctest -N` / `ctest -R p0_1_engine_fail_closed` 列出）。
- `COMMAND` 为 `bash` + 仓库内该脚本的绝对/源码树路径（指向 `scripts/test_p0_1_engine_fail_closed.sh`）。
- `WORKING_DIRECTORY` 为 `${PROJECT_ROOT_DIR}`（与现有 gtest `add_test` 一致），保证脚本相对路径、`conf/`、`weights/`、`scripts/` 解析正确。
- 设 `RUN_SERIAL TRUE`（或等价互斥资源锁）：脚本会临时改写/恢复 `scripts/fetch_weights.py`（及可能的 `verify_deployment.sh`），并行 `ctest -j` 会互相踩文件。
- 不打 `valgrind`/`sanitizer` label；不改脚本正文的 PASS/FAIL/SKIP 与退出码；不为该用例伪造「无 GPU 即 SKIPPED」的 CTest 状态。
- 代价：只动测试注册一层；默认 `check`/ctest 多跑一轮黑盒（无 GPU 时 AC8 仍 skip，整体仍可绿）；需串行约束，避免并行踩 stub。

### 否决

1. **只改 CI workflow / 文档手跑提醒，不 `add_test`**  
   代价低，但不满足「默认 ctest 出现该脚本对应用例」；小团队只跑 `check`/默认 ctest 仍漏掉。
2. **把脚本改写成 gtest/`system()` 包装，或改脚本退出码对接 `SKIP_RETURN_CODE`**  
   改动面与维护成本更高；易削弱已有 AC；与「无 GPU 时 SKIP 的退出码约定不变」冲突。不采用。

## 改动面

- `test/CMakeLists.txt`：新增上述 `add_test` + `set_tests_properties`（至少 `RUN_SERIAL`）。
- 可选且非必须：若仓库内已有「如何跑默认 ctest」说明且明确列举用例来源，可一句指向该用例名；**不是**本条验收前提，RD 不得借文档扩大范围。

## 明确不改

- `src/` 全部。
- `scripts/test_p0_1_engine_fail_closed.sh` 的 AC 逻辑、`FAILS`/`exit 0|1` 约定、AC8 skip 语义。
- `scripts/mortredctl_*.sh`、`product_static_bs1.py`、`pack_trt.py`、`check_consistency.py` 等产品/运维脚本行为（本条只调用既有脚本）。
- `check` 目标定义（仍是 `ctest --output-on-failure -LE valgrind`）与 valgrind 门禁开关。
- 产品 toml / engine 真源 / HTTP·tensor 契约 / gpu|cpu profile / TensorRT prepare 流程。
- P1-1、P1-2 已合入内容。

## 契约

- **HTTP / 推理 tensor / 产品 toml**：不变。
- **测试契约**：默认 ctest 目录中新增可筛选用例名 `p0_1_engine_fail_closed`；执行体仍是既有黑盒脚本；成功判定仍是脚本 `exit 0`，失败 `exit 1`。
- **CTest 状态**：无 GPU 时脚本内 AC8 打印 `[SKIP]` 后若其余 AC 通过，ctest 看到的是 **Passed（退出码 0）**，不是把整用例标成 CTest Skipped；禁止用改退出码换 Skipped。

## 性能与资源

- 不启 catalog/gateway/supervisor 常驻进程，不要求生成/加载 TensorRT engine，不额外占显存。
- AC3–AC7 会短暂触碰本地 engine 路径文件与 stub 脚本；结束后须由脚本既有 `trap` 恢复；勿与并行改同一脚本的测试同跑（故 `RUN_SERIAL`）。
- 时长：秒级到分钟级（取决于本机 python/init 路径），相对全量 gtest 可接受；不得因此关掉 AC 断言。

## 验收标准

1. **AC1（用例可列出）**：在已 configure 的 build 目录执行 `ctest -N`（或 `ctest -N -R p0_1_engine_fail_closed`），输出中出现测试名 `p0_1_engine_fail_closed`。
2. **AC2（进入默认门、非 valgrind）**：同一 build 下，默认 `check` 所跑集合（等价 `ctest --output-on-failure -LE valgrind`）会执行该用例；`ctest -N -L valgrind` 列表中**不**出现该名。
3. **AC3（tests-only 与 FULL 均可注册）**：tests-only 与 FULL 各至少一次 configure 后，均可在对应 build 的 `ctest -N` 中看到该用例名。
4. **AC4（退出码约定不变·无 GPU）**：在无 GPU/`trtexec`/所需 onnx、因而 AC8 走脚本内 skip 的 WSL 环境，直接跑脚本与经 ctest `-R p0_1_engine_fail_closed` 跑该用例，在其余 AC 本应通过的前提下，进程退出码均为 `0`；输出仍可见 AC8 的 `[SKIP]`，且不得把「整用例失败」当成 skip 成功。
5. **AC5（失败仍红）**：故意破坏脚本可判定项之一（例如临时破坏 AC1 路径一致性所依赖的配置一致性，测完恢复），经 ctest 跑该用例退出码为 `1`（或使该 ctest 失败）；恢复后应回到 AC4 的绿。
6. **AC6（未削弱既有验收、未碰产品代码）**：diff 不含 `src/`；脚本 AC1–AC7 断言语义与退出码约定相对合入前无弱化；不得以 `--help` 或仅「脚本存在」代替挂测成功。

## 测试义务

- QA 按 AC1–AC6 做黑盒：列名、默认 `check`/`-LE valgrind` 路径、tests-only 与 FULL 各一次列出、WSL 无 GPU 退出码 0 + AC8 skip、人为制造失败后 ctest 红再恢复、确认无 `src/` 改动且脚本约定未改。
- QA 不重测 P0-1 产品改动本身，只确认挂测后默认 ctest 会跑到该回归。
- 并行 `ctest -j` 下若未串行导致 stub 残留，判实现未满足 `RUN_SERIAL`（或等价）义务。

## 自测命令

均在仓库根、WSL 执行；`BUILD` 换成实际 build 目录（tests-only 或 FULL 皆可；AC3 要求两种各做一遍列出）。

```bash
# 1) 列出用例（期望：输出含 p0_1_engine_fail_closed；ctest 自身退出码 0）
ctest --test-dir "$BUILD" -N -R '^p0_1_engine_fail_closed$'
# 期望退出码: 0

# 2) 确认不在 valgrind 标签集（期望：匹配列表为空或不含该名；ctest -N 退出码 0）
ctest --test-dir "$BUILD" -N -L valgrind -R '^p0_1_engine_fail_closed$'
# 期望退出码: 0，且无该测试名

# 3) 手跑脚本（无 GPU 时 AC8 skip，FAILS=0）
bash scripts/test_p0_1_engine_fail_closed.sh
# 期望退出码: 0

# 4) 经默认过滤跑该用例
ctest --test-dir "$BUILD" --output-on-failure -LE valgrind -R '^p0_1_engine_fail_closed$'
# 期望退出码: 0

# 5) 默认 check（需已能 build check 依赖；耗时更长）
cmake --build "$BUILD" --target check
# 期望退出码: 0，且日志/LastTest 中执行过 p0_1_engine_fail_closed
```

（AC5 人为失败用毕必须恢复工作树；自测记录须体现失败时 ctest 非 0、恢复后为 0。）

## RD 不得自行决定

- 用例正式名必须是 `p0_1_engine_fail_closed`（除非本方案后续升版改名并同步验收）。
- 不得修改 `scripts/test_p0_1_engine_fail_closed.sh` 的退出码/AC8 skip 约定，不得引入 `SKIP_RETURN_CODE` 改变「无 GPU 仍可整体 exit 0」。
- 不得改 `src/`，不得把该门禁收到仅 CI job、仅 FULL、或 `valgrind` label 下。
- `WORKING_DIRECTORY` 必须是仓库根 `${PROJECT_ROOT_DIR}`；必须对 stub 写回风险设 `RUN_SERIAL`（或书面等价互斥）。
- 不得用 `--help`、空包装脚本、或删减 AC 断言冒充挂测完成。

## 签字
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| cv-pm | 同意 | 无 | v1 | `cowork/ledger/round-004/issues/p1-3/03-discussion/v1-sign-pm.md` |
| cv-rd | 同意 | 无 | v1 | `cowork/ledger/round-004/issues/p1-3/03-discussion/v1-sign-rd.md` |
| cv-tech-lead | 同意 | 无 | v1 | `cowork/ledger/round-004/issues/p1-3/03-discussion/v1-sign-tech-lead.md` |
