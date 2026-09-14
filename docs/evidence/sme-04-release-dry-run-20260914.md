# SME-04 evidence — release dry-run (WSL)

- **Date:** 2026-09-14
- **Branch / HEAD:** `fix/sme-04-release-dry-run` @ `1ed92a4`
- **Host:** WSL (`DESKTOP-B8MJFUS`), checkout `/mnt/g/Codex/mortred_model_server`

## Goal

Local rehearsal (no GHCR push, no GitHub Release) of:

1. `release.yml` GHCR IMAGE lowercasing
2. tarball + basename `.sha256` (`sha256sum -c`)
3. bootstrap-style checksum mismatch refuse (+ WARN branch present for missing `.sha256`)

## Commands and results

```text
bash scripts/release_dry_run.sh; echo dry_exit=$?
→ release_dry_run: ALL OK
→ dry_exit=0

python3 scripts/check_consistency.py; echo pass_exit=$?
→ Repository consistency check passed.
→ pass_exit=0
```

### Negative gate (deliberately break lowercase)

Replaced first `tr '[:upper:]' '[:lower:]'` with `cat` in `.github/workflows/release.yml`:

```text
python3 scripts/check_consistency.py; echo neg_exit=$?
→ release.yml: expected >=2 lowercase IMAGE constructions (found 1)
→ neg_exit=1
```

Restored file:

```text
restore_exit=0
→ Repository consistency check passed.
```

## Outcome

**PASS.** SME-04 acceptance met: dry-run script green; consistency gate catches a broken GHCR lowercase construction.
