# SME-05 evidence — pack_file → apply_pack (WSL)

- **Date:** 2026-09-14
- **Branch / HEAD:** `fix/sme-05-pack-file-apply` @ `f32e47a`
- **Host:** WSL (`DESKTOP-B8MJFUS`), `/mnt/g/Codex/mortred_model_server`

## Goal

`[supervisor].pack_file` in `conf/mortred.toml` must call `ControlConfig::apply_pack` when `MORTRED_PACK` is unset. Env/CLI still wins via `resolve_pack_path`.

## Unit tests

```text
./_bin/control_config_unittest --gtest_filter='*pack*'
→ 6 tests PASSED (incl. resolve_pack_path_prefers_override_then_pack_file, load_parses_pack_file)
→ ut_exit=0
```

## Supervisor smoke

Rebuilt `mortred-supervisor.out` with full-cpu preset. Enabled:

```toml
pack_file = "conf/packs/demo.toml"
```

`unset MORTRED_PACK`, `MORTRED_PROFILE=cpu`, started supervisor:

```text
mortred-supervisor: pack /mnt/g/Codex/mortred_model_server/conf/packs/demo.toml (autostart 1 model(s))
mortred-supervisor listening on http://127.0.0.1:8787 (managed servers: 2, auth enabled, expose=loopback)
```

## Outcome

**PASS.** Config and behavior aligned; docs/comments updated accordingly.
