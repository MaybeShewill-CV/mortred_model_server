# SME-13 evidence — drop zombie web_console/ServerManager claims

- **Date:** 2026-09-14
- **Branch / HEAD:** `fix/sme-13-zombie-comments` @ `8d96fbf`
- **Host:** docs/source review (WSL/Mac)

## Checks

```text
CMakeLists.txt — _bin/_lib comment no longer cites web_console/ServerManager
test/ready_probe_unittest.cc — header describes supervisor/control ready_probe
scripts/clean_artifacts.sh — removed src/apps/web_console/backend/build

grep -rn 'web_console\|ServerManager' (excl. .git/build/3rd_party)
→ only CHANGELOG Fixed entry + sme-oob-checklist (no live source claims)
```

## Outcome

**PASS.** Last SME-oob checklist item; source tree has no zombie claims that those
removed apps still exist.
