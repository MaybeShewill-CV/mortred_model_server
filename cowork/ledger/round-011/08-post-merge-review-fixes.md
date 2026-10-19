# 合入后的项目级复核（QA 第二轮）与修复

- 复核对象: `sub-agent-co-work @ 35336365`（`636a3990` 已合入 `fix/p1-5-9-metadata`）
- 复核结论: **fail**（本分支定向验收在 HEAD 仍全绿，但放大到全项目后发现 3 条）
- 修复提交: **`82e70987`**（5 文件 +32/−6）
- 修复后复核: 带断言脚本 `logs/_qa2_allfixed.sh` → **`SUMMARY FAILS=0`**

## 三条发现（均由我先独立复现，再修）

| # | 级别 | 问题 | 我的复现 | 修法 | 现状 |
| --- | --- | --- | --- | --- | --- |
| D1 | P1 | **P1-6 未收口**：三处主文档仍教 `echo -n "your-secret-key" \| sha256sum`（按文档生成的整定会被新 guard 拒绝）；`mint_api_key.sh` 在 docs 里不可发现 | `grep -rn your-secret-key docs/` → 3 处；`git grep -l mint_api_key -- docs` → 空 | 三处改为用 `scripts/mint_api_key.sh`；`api-keys.md` 增加 helper 优先用法 + guard/doctor 说明；中英同步 | `grep -rc your-secret-key docs/` → **0**；`git grep -l mint_api_key -- docs` → **3** |
| D2 | P1（**预存在**，非本分支引入） | 项目自带 CI 门禁 `scripts/check_consistency.py` 在 HEAD **确定性退出 1** | `python3 scripts/check_consistency.py` → 1；`python3 scripts/calibrate_pack.py --self-test` → 1，报 `self-test: ineligible row produced pack fields {}` | 根因是 `occupancy_fields_from_report()` 在**没有任何 eligible 行**时仍无条件写入 `host_memory_total_mib`（读 `/proc/meminfo`），而自测（`:1967`）断言"无 eligible 行时 pack 字段为空"、下一条还要求"pack 文本不变"。判定为**产品侧越界写**：没有合格行时直接返回空 pack 字段（3 行 early return），保持"不合格就不碰 pack"的语义 | `calibrate --self-test` → **0**；`check_consistency.py` → **0**（`Repository consistency check passed.`） |
| D3 | P2 | 我新写的 `mint_api_key.sh:106` 声称 gateway **热加载** `conf/api_keys.toml`，与实现/文档相反 | `grep -n hot-reloads scripts/mint_api_key.sh`；`git grep -n 'reload()' -- src` 只有声明/定义（无调用点）；docs 明确 no hot reload；`gateway_app.cpp:11` 注释 `config reload = restart` | 文案改为 "There is **NO** hot reload: restart the gateway child (`POST /api/v1/servers/__gateway/restart`)" | 文案已改；`mint --self-test` 仍 `SUMMARY FAILS=0` |

## 对既有说法的两点更正

1. **P1-6 在合入时并未真正收口**：`07-fix-p1-5-p18-closure.md` 的 P1-6 行说"示例改为不要哈希人类口令"只对 `conf/api_keys.toml.example` 成立；三处主文档直到 `82e70987` 才同步。已在该文件加"更正"批注。
2. **"编排方合入后复核 SUMMARY FAILS=0"不能推出 CI 门禁为绿**：那次复核只覆盖 4 个脚本自测与 8 个测试二进制，**不含** `check_consistency.py`。D2 之前一直是红的，只因 workflow 仅在 push main / PR / nightly 触发，本分支从未被 CI 跑过。

## 仍未做（A4）

全量 101 项 `check` 未跑完；`model_golden_test` 因本机 20 GB 内存会 OOM 而被排除。因此**"整套全绿"仍是未证明状态**。若要做，需要在串行/分片或更大内存条件下跑，并把 `model_golden_test` 单独隔离。
