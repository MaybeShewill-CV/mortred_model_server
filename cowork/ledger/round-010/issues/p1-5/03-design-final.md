# 技术方案

- 问题: P1-5
- 版本: v1（编排方决策版）
- 对应列表: cowork/ledger/round-010/02-p0-p1-final.md P1-5

## 边界

只改 `model_run_timeout<=0` 的处置（`src/server/server_runtime_config.h:151-157`）与其下游后果（`base_server_impl.h` 里无界 checkout / 不创建请求定时器 / `_m_waiting_jobs` 不回落）。不改 `receive_timeout`、`stuck_worker_*`、`max_queue_depth` 的语义。

## 做法

### 推荐

与同文件 `receive_timeout<=0`（`:138-142`）同构：`LOG(WARNING)` + **夹到一个有界值**，但**不能夹到 500ms 的默认值**——`conf/server/**` 里有 5 份配置显式写 `model_run_timeout=0`（`sam_amg`、`ddpm`、`ddim`、`ldm`、`cls_cond_ddim`），夹到 500ms 会把它们变成 504 机器。

具体：

1. `server_runtime_config.h`：新增常量 `k_model_run_timeout_clamp_ms = 300000`（5 分钟，与既有 `async_timeout` 默认值同量级）。`cfg.model_run_timeout <= 0` 时置为该值并 WARNING，措辞写明生效值（`"model_run_timeout <= 0: clamping to 300000ms; a request always gets a deadline and a timer"`）。
2. 同一改动把 5 份 conf 的 `model_run_timeout=0` 改成显式 `300000`，让"慢模型需要长超时"这一意图留在配置里而不是藏在夹值逻辑里。
3. `docs/about_model_server_configuration.md`(+`.zh-cn.md`)：删掉 `model_run_timeout` 的"-1/0 表示无限大"表述，改为"必须 > 0；`<=0` 会被夹到 300000ms 并告警"。
4. 测试：`server_runtime_config_unittest` 覆盖 `0` / `-1` / 缺键三态（缺键仍取默认 500）；`server_e2e_contract_test` 增加可判别用例：`model_run_timeout=0` + `fake_delay_ms=1000` → 服务正常启动且请求得到 **504**（当前实现会返回 200，因为无 deadline、无定时器）。

### 否决

- **夹到 `defaults.model_run_timeout`（500ms）**：技术负责人线的原推荐，会直接打断 5 个已上线 id。
- **拒绝启动**：5 份合法配置起不来，且与 `receive_timeout` 的既有处置不一致。
- 只加 WARNING 不改值：就是当前缺陷本身。
- 把 0 解释为"很大但有限"的未文档化魔法值。

## 改动面

- `src/server/server_runtime_config.h`（夹值 + 常量 + WARNING）
- `conf/server/**` 里 5 份 `model_run_timeout=0` → `300000`
- `docs/about_model_server_configuration.md`、`docs/about_model_server_configuration.zh-cn.md`
- `test/server_runtime_config_unittest.cc`、`test/server_e2e_contract_test.cc`

## 明确不改

- `server_config_schema.h` 的键白名单（该键只做类型校验，无下界，无需改）
- `base_server_impl.h` 的超时/定时器代码（它本来就按 `> 0` 工作）
- `stuck_worker_action`、`receive_timeout`、packing/max_batch_size

## 契约

- `model_run_timeout` 生效值恒 > 0：每个请求都有 deadline、都创建请求定时器、`_m_waiting_jobs` 必然回落；不存在"无界等待"配置。
- `<=0` 被夹到 300000ms 并在启动日志中告警，生效值可从日志读出。

## 性能与资源

只影响配置解析。5 个慢模型（SAM AMG / diffusion 家族）的超时从"无限"变为"5 分钟"；正常推理远低于该值，只在真挂死时触发 504 并释放 worker。

## 验收标准

- A1 `./_bin/server_runtime_config_unittest` 退出 0，含 `=0` 与 `=-1` → 300000、缺键 → 500 三态用例。
- A2 `./_bin/server_e2e_contract_test` 退出 0，含新用例：`model_run_timeout=0` + `fake_delay_ms=1000` → 504，且服务**正常启动**（不是拒绝启动）。
- A3 `grep -rn 'model_run_timeout=0' conf/server | wc -l` = 0，且 `grep -rn 'model_run_timeout' conf/server | wc -l` 仍为 5 份配置均显式有界。
- A4 请求超时后 `/metrics` 的 `mortred_waiting_jobs` 回落到 0（不残留）。
- A5 提交只含上述 6 类文件；不含 `cowork/ledger/`、构建产物、`src/common/mortred_version.h`。

## 测试义务

- 上述 5 条，给出真实命令与退出码。
- 白盒：确认夹值发生在 `parse_server_runtime_config` 内、且 `apply_runtime_config` 读到的是夹后值。
- 回归：`receive_timeout` 既有用例与"缺键取默认"用例仍绿。

## 自测命令

```bash
cd /mnt/g/Codex/mortred_wt_r10b
cmake --build build --target server_runtime_config_unittest server_e2e_contract_test -j8   # 期望 0
./_bin/server_runtime_config_unittest ; echo $?        # 期望 0
./_bin/server_e2e_contract_test --gtest_filter='*timeout*' ; echo $?   # 期望 0
grep -rn 'model_run_timeout=0' conf/server | wc -l     # 期望 0
```

## RD 不得自行决定

- 夹值大小（固定 300000ms；如需其它值须回到编排方）；是夹值还是拒绝启动（选夹值）；是否为慢模型单独放宽默认值。
