#!/usr/bin/env python3
"""Fail closed if the container path drifts from the documented GPU default.

Catches, without booting an image:
  * docker_entrypoint.sh missing a shebang (exec-form ENTRYPOINT cannot run it)
  * compose gpu service with an empty/missing build target (Compose then
    builds Docker's last stage, which must stay mortred-gpu)
  * a new CPU (or other) FROM appended after mortred-gpu, silently changing
    `docker build .`
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "scripts" / "docker_entrypoint.sh"
DOCKERFILE = ROOT / "Dockerfile"
COMPOSE = ROOT / "docker-compose.yml"
EXEC_ENTRYPOINT = 'ENTRYPOINT ["/opt/mortred/scripts/docker_entrypoint.sh"]'
FROM_RE = re.compile(
    r"^FROM\s+(\S+)(?:\s+AS\s+(\S+))?", re.IGNORECASE
)


def fail(msg: str) -> None:
    print("[FAIL] %s" % msg, file=sys.stderr)
    raise SystemExit(1)


def assert_shebang() -> None:
    first = ENTRYPOINT.read_text(encoding="utf-8").splitlines()[0].strip()
    if first != "#!/usr/bin/env bash":
        fail(
            "scripts/docker_entrypoint.sh must start with '#!/usr/bin/env bash' "
            "(got %r). exec-form ENTRYPOINT cannot run a script without a shebang."
            % first
        )
    print("[ok] docker_entrypoint.sh shebang")


def dockerfile_from_stages() -> list[tuple[str, str | None]]:
    stages: list[tuple[str, str | None]] = []
    for raw in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = FROM_RE.match(line)
        if m:
            stages.append((m.group(1), m.group(2)))
    return stages


def assert_dockerfile() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    if EXEC_ENTRYPOINT not in text:
        fail("Dockerfile is missing exec-form %s" % EXEC_ENTRYPOINT)
    stages = dockerfile_from_stages()
    if not stages:
        fail("Dockerfile has no FROM lines")
    image, alias = stages[-1]
    if image != "runtime" or alias != "mortred-gpu":
        fail(
            "Dockerfile last FROM must be 'FROM runtime AS mortred-gpu' "
            "(Docker's default target). Last stage is FROM %s AS %s."
            % (image, alias)
        )
    aliases = {a for _, a in stages if a}
    for required in ("runtime", "mortred-cpu", "mortred-gpu"):
        if required not in aliases:
            fail("Dockerfile is missing named stage %s" % required)
    print("[ok] Dockerfile last stage is mortred-gpu")


def compose_build_targets(path: pathlib.Path) -> dict[str, str]:
    """Read services.*.build.target without PyYAML (stdlib-only)."""
    targets: dict[str, str] = {}
    current: str | None = None
    in_build = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        token = line.strip()
        if indent == 2 and token.endswith(":"):
            current = token[:-1]
            in_build = False
            continue
        if current is None:
            continue
        if indent <= 2:
            current = None
            in_build = False
            continue
        if indent == 4 and token == "build:":
            in_build = True
            continue
        if in_build and indent == 4:
            in_build = False
        if in_build and indent >= 6 and token.startswith("target:"):
            value = token.split(":", 1)[1].strip().strip("\"'")
            targets[current] = value
    return targets


def assert_compose() -> None:
    targets = compose_build_targets(COMPOSE)
    gpu_target = targets.get("mortred", "")
    cpu_target = targets.get("mortred-cpu", "")
    if gpu_target != "mortred-gpu":
        fail(
            "docker-compose.yml service mortred build.target must be "
            "'mortred-gpu' (got %r). Empty target builds the last Dockerfile "
            "stage and previously silently produced the CPU image."
            % gpu_target
        )
    if cpu_target != "mortred-cpu":
        fail(
            "docker-compose.yml service mortred-cpu build.target must be "
            "'mortred-cpu' (got %r)" % cpu_target
        )
    print("[ok] compose gpu target=mortred-gpu, cpu target=mortred-cpu")


def main() -> int:
    assert_shebang()
    assert_dockerfile()
    assert_compose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
