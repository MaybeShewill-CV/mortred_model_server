#!/usr/bin/env python3
"""Read-only check for the README GPU source-build command sets.

Compares the GPU command set extracted from the English
``### Building from source`` first bash fence with the set from the Chinese
``#### 路径 B：full build`` bash fence. Also checks the two cmake preset
lines, ``CMakePresets.json`` preset names, ``verify_deployment.sh --basic``,
and the two ``--cpu`` commands.

The source-build section starts on the line after the heading and stops
before the next heading outside a code fence. A heading is a line that,
after leading whitespace is removed, starts with ``# ``. Lines that start
with a fence marker toggle inside and outside. Lines inside a fence,
including ``# dependencies``, ``# Ubuntu 22.04``, and ``# 1. 安装 vcpkg``,
do not end the section.

This script only reads ``README.md``, ``README.zh-cn.md``, and
``CMakePresets.json``. It does not run install_deps.sh, cmake, or Docker.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EN_README = "README.md"
ZH_README = "README.zh-cn.md"
PRESETS_FILE = "CMakePresets.json"

EN_HEADING = "### Building from source"
ZH_HEADING = "### 从源码构建"
ZH_PATH_B = "#### 路径 B：full build"

GPU_COMMANDS = frozenset(
    {
        "./scripts/install_deps.sh --all",
        "sudo ./scripts/install_deps.sh --nvidia",
        "./scripts/install_deps.sh --check",
        "cmake --preset full && cmake --build --preset full",
    }
)

PRESET_COMMANDS = (
    "cmake --preset full && cmake --build --preset full",
    "cmake --preset full-cpu && cmake --build --preset full-cpu",
)

CPU_COMMANDS = (
    "./scripts/install_deps.sh --cpu --all",
    "./scripts/install_deps.sh --cpu --check",
)

VERIFY_COMMAND = "./scripts/verify_deployment.sh --basic"

PRESET_GROUPS = ("configurePresets", "buildPresets")
PRESET_NAMES = ("full", "full-cpu")


def read_lines(path: Path) -> list[str]:
    # splitlines() treats \r\n, \n, and a bare \r as line breaks.
    return path.read_bytes().decode("utf-8").splitlines()


def normalize(line: str) -> str:
    if line.endswith("\r"):
        line = line[:-1]
    comment = line.find(" #")
    if comment >= 0:
        line = line[:comment]
    return line.strip()


def dropped(text: str) -> bool:
    return text == "" or text.startswith("#")


def section_after(lines: list[str], heading: str) -> list[str] | None:
    start = None
    for index, line in enumerate(lines):
        if line.strip() == heading:
            start = index + 1
            break
    if start is None:
        return None
    body: list[str] = []
    in_fence = False
    for line in lines[start:]:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            body.append(line)
            continue
        if not in_fence and line.lstrip().startswith("# "):
            break
        body.append(line)
    return body


def bash_fence_after(section: list[str], marker: str | None) -> list[str] | None:
    """Return one bash fence body.

    ``marker is None`` selects the first bash fence. Otherwise the fence is
    the first bash fence after the section line equal to ``marker``.
    """
    seen_marker = marker is None
    in_fence = False
    info = ""
    body: list[str] = []
    for line in section:
        if not seen_marker:
            if line.strip() == marker:
                seen_marker = True
            continue
        if line.lstrip().startswith("```"):
            if not in_fence:
                in_fence = True
                info = line.lstrip()[3:].strip()
                body = []
            else:
                if info == "bash":
                    return body
                in_fence = False
                info = ""
                body = []
            continue
        if in_fence:
            body.append(line)
    return None


def gpu_set(fence_body: list[str]) -> set[str]:
    found: set[str] = set()
    for line in fence_body:
        text = normalize(line)
        if dropped(text):
            continue
        if text in GPU_COMMANDS:
            found.add(text)
    return found


def section_texts(section: list[str]) -> set[str]:
    found: set[str] = set()
    for line in section:
        text = normalize(line)
        if dropped(text):
            continue
        found.add(text)
    return found


def check_gpu_set(errors: list[str], label: str, found: set[str]) -> None:
    for command in sorted(GPU_COMMANDS - found):
        errors.append(f"{label} gpu set missing: {command}")


def check_present(errors: list[str], label: str, present: set[str], commands: tuple[str, ...]) -> None:
    for command in commands:
        if command not in present:
            errors.append(f"{label} section missing: {command}")


def check_readme(
    errors: list[str],
    filename: str,
    heading: str,
    fence_marker: str | None,
    require_verify: bool,
) -> None:
    path = ROOT / filename
    if not path.is_file():
        errors.append(f"{filename} gpu set missing: file")
        return
    try:
        lines = read_lines(path)
    except (OSError, UnicodeError) as exc:
        errors.append(f"{filename} gpu set missing: {exc}")
        return
    section = section_after(lines, heading)
    if section is None:
        errors.append(f"{filename} gpu set missing heading: {heading}")
        return
    fence = bash_fence_after(section, fence_marker)
    if fence is None:
        where = "first bash fence" if fence_marker is None else fence_marker
        errors.append(f"{filename} gpu set missing bash fence: {where}")
        found: set[str] = set()
    else:
        found = gpu_set(fence)
    check_gpu_set(errors, filename, found)
    present = section_texts(section)
    check_present(errors, filename, present, PRESET_COMMANDS)
    check_present(errors, filename, present, CPU_COMMANDS)
    if require_verify and VERIFY_COMMAND not in present:
        errors.append(f"{filename} section missing: {VERIFY_COMMAND}")


def check_presets(errors: list[str]) -> None:
    path = ROOT / PRESETS_FILE
    names_by_group: dict[str, set[str]] = {group: set() for group in PRESET_GROUPS}
    try:
        document = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"{PRESETS_FILE} configurePresets missing name: full ({exc})")
        document = None
    if isinstance(document, dict):
        for group in PRESET_GROUPS:
            items = document.get(group, [])
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    names_by_group[group].add(item["name"])
    for group in PRESET_GROUPS:
        for name in PRESET_NAMES:
            if name not in names_by_group[group]:
                errors.append(f"{PRESETS_FILE} {group} missing name: {name}")


def main() -> int:
    errors: list[str] = []
    check_readme(errors, EN_README, EN_HEADING, None, True)
    check_readme(errors, ZH_README, ZH_HEADING, ZH_PATH_B, False)
    check_presets(errors)
    if errors:
        for message in errors:
            print(message, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
