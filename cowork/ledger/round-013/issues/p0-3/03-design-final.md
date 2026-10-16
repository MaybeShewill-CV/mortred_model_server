# Design P0-3 v1 (user-locked)
## Goal
http_infer_rps.py default warmup works; self-test after parse; smoke covers default path.
## Fix
parse_duration allows >=0; parse before --self-test; assert 0s==0.
