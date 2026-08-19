"""Minimal TOML loading shared by repo scripts.

Prefers the stdlib `tomllib` (Python >= 3.11, e.g. ubuntu-24.04) and falls
back to a small regex parser sufficient for this repo's flat, quoted-string
config files (ubuntu-22.04 ships Python 3.10 without tomllib).
"""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    tomllib = None

_SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_KV_RE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"([^"]*)"\s*(?:#.*)?$')


def load_toml(path: str | Path) -> dict:
    path = Path(path)
    if tomllib is not None:
        with path.open("rb") as f:
            return tomllib.load(f)

    result: dict = {}
    section = result
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            match = _SECTION_RE.match(line)
            if match:
                section = result.setdefault(match.group(1), {})
                continue
            match = _KV_RE.match(line)
            if match:
                section[match.group(1)] = match.group(2)
    return result
