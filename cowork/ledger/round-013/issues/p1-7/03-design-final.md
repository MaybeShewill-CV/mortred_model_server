# Design P1-7 v1 (user-locked)
## Goal
SIGKILL of supervisor reaps GPU children; orphans self-exit.
## Fix
1. deploy/mortred-supervisor.service KillMode=control-group (systemd kills whole cgroup).
2. Keep PR_SET_PDEATHSIG(SIGINT); document that control-group is the hard-kill path.
3. Optional: children already in supervisor PG — note in unit comment.
