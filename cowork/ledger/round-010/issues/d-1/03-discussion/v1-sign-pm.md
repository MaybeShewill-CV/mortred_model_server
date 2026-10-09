# 第3步签字

- 问题: D-1
- 方案版本: v1
- 结论: 不同意
- 条件: 验收第 4 条改成现有用例已经断言的结果。`autostart_models_failed_on_spawn_failure` 仍通过，并只判定 `missing_model` 的 `pid <= 0`、`reason` 含 `missing_model` 与 `failed`、`models_failed` 为 true。`empty_server_uri_fails_readiness_immediately` 仍通过，判定 `state` 为 `failed` 且 `error` 含 `server_uri`。`autostart_skips_models_when_gateway_deadline_missed` 仍通过，判定 `models_skipped` 为 true 且 `fake_model_server` 的 `pid <= 0`。若要判定 spawn 失败 child 的 `state` 为 `failed`，或网关 deadline 时该模型的 `state` 不是 `starting`，把这两句写成上述用例里的 `status().state` 断言。契约里「已 `running` 的兄弟 child 在本波因其他 child 触发 deadline 时仍为 `running`」补成编号验收，用该兄弟的 `status().state` 判定。
- 理由: 验收 1、2、3、5 可以用 `status`、`autostart_gate` 或现有 `autostart_healthy_models_not_failed` 判定。第 4 条写「失败 child 为 `failed`」，但 `autostart_models_failed_on_spawn_failure` 不断言 child 的 `state`。同条写网关 deadline 时该模型不是 `starting`，但 `autostart_skips_models_when_gateway_deadline_missed` 只断言 `pid <= 0`。这两句不能由现有单测通过与否判定。兄弟 child 保持 `running` 写在契约里，没有编号验收；现有 `autostart_healthy_models_not_failed` 在 deadline 前整波就绪，也判定不了这条。
