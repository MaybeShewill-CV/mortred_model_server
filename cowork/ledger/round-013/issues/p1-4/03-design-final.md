# Design P1-4 v1 (user-locked)
## Goal
Relative MORTRED_CONTROL_CONFIG resolves against project_root before child inject/chdir.
## Fix
1. supervisor_app: if env path is relative, absolutize via project_root.
2. supervisor spawn: absolutize control_config against project_root before inject.
3. gateway_app: if env path relative, resolve via MORTRED_PROJECT_ROOT / resolve_project_root.
