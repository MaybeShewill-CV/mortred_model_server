#!/usr/bin/env python3
"""Merge per-id pack files into the one pack file prepare/calibrate accept.

mortredctl prepare, scripts/prepare_pack.sh, scripts/pack_trt.py, and
MORTRED_PACK each take a single --pack path. Product coverage lives in
conf/packs/<id>.toml (one [pack.<id>] section). Co-starting several catalog
ids means composing those files:

  python3 scripts/compose_packs.py \\
      conf/packs/yolov8s.toml conf/packs/libface.toml \\
      -o /tmp/yolov8s-libface.toml
  mortredctl prepare --pack /tmp/yolov8s-libface.toml

Sections stay in argument order. A repeated [pack.<id>] is an error.
[pack] keys merge when the assignment text matches and conflict otherwise.
Output under conf/packs/ or conf/server/ is refused; calibrate --write-pack
has the same ban, so the composed file belongs on the machine (for example
/tmp/combined.toml).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ComposeError(Exception):
    """The inputs cannot be merged into one pack file."""


def parse_pack(path: Path) -> list[tuple[str, list[str]]]:
    """Ordered tables. '' is the global [pack] table; other names are ids.

    Body entries are stripped ``key = value`` lines. Comments are dropped.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ComposeError(f"cannot read {path}: {exc}") from exc
    sections: list[tuple[str, list[str]]] = []
    current: str | None = None
    body: list[str] = []
    seen_keys: set[str] = set()

    def close() -> None:
        nonlocal current, body, seen_keys
        if current is not None:
            sections.append((current, body))
        current = None
        body = []
        seen_keys = set()

    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            close()
            if stripped == "[pack]":
                name = ""
            elif stripped.startswith("[pack.") and stripped.endswith("]"):
                name = stripped[len("[pack.") : -1].strip()
                if not name:
                    raise ComposeError(f"{path}:{lineno}: empty pack id")
            else:
                raise ComposeError(f"{path}:{lineno}: unsupported table {stripped}")
            if any(existing == name for existing, _ in sections):
                label = "[pack]" if name == "" else f"[pack.{name}]"
                raise ComposeError(f"{path}:{lineno}: duplicate {label}")
            current = name
            continue
        if current is None or not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ComposeError(f"{path}:{lineno}: expected key = value")
        key = stripped.split("=", 1)[0].strip()
        if not key:
            raise ComposeError(f"{path}:{lineno}: empty key")
        if key in seen_keys:
            raise ComposeError(f"{path}:{lineno}: duplicate key {key}")
        seen_keys.add(key)
        body.append(stripped)
    close()
    if not any(name for name, _ in sections):
        raise ComposeError(f"{path}: no [pack.<id>] section")
    return sections


def compose_text(paths: list[Path]) -> str:
    """Render one pack document from ``paths`` (already-existing files)."""
    if not paths:
        raise ComposeError("at least one pack file is required")
    merged: dict[str, tuple[Path, list[str]]] = {}
    order: list[str] = []
    global_lines: list[str] = []
    global_from: dict[str, tuple[Path, str]] = {}
    for path in paths:
        if not path.is_file():
            raise ComposeError(f"pack not found: {path}")
        for name, body in parse_pack(path):
            if name == "":
                for line in body:
                    key = line.split("=", 1)[0].strip()
                    prev = global_from.get(key)
                    if prev is not None and prev[1] != line:
                        raise ComposeError(
                            f"[pack] {key} conflicts: {prev[0]} vs {path}"
                        )
                    if prev is None:
                        global_from[key] = (path, line)
                        global_lines.append(line)
                continue
            if name in merged:
                raise ComposeError(
                    f"duplicate pack id {name}: {merged[name][0]} and {path}"
                )
            merged[name] = (path, body)
            order.append(name)
    lines = [
        "# Composed pack for one --pack file (prepare, calibrate, doctor, MORTRED_PACK).",
        "# Inputs:",
    ]
    for path in paths:
        lines.append("#   %s" % path.as_posix())
    lines.append(
        "# Duplicate ids are rejected. calibrate --write-pack refuses conf/packs/."
    )
    lines.append("")
    if global_lines:
        lines.append("[pack]")
        lines.extend(global_lines)
        lines.append("")
    for name in order:
        lines.append("[pack.%s]" % name)
        lines.extend(merged[name][1])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def refuse_repo_output(path: Path) -> None:
    """calibrate --write-pack refuses these trees; compose does too."""
    resolved = path.resolve()
    for banned in (ROOT / "conf" / "packs", ROOT / "conf" / "server"):
        try:
            resolved.relative_to(banned.resolve())
        except ValueError:
            continue
        raise ComposeError(
            "refusing to write %s; compose into a machine-local file "
            "such as /tmp/combined.toml" % path
        )


def compose(paths: list[Path], output: Path) -> str:
    """Write ``output`` and return the text that was written."""
    refuse_repo_output(output)
    text = compose_text(paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return text


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a = root / "a.toml"
        b = root / "b.toml"
        a.write_text(
            "\n".join(
                [
                    "# seed a",
                    "[pack.yolov8s]",
                    "worker_nums = 1",
                    'model_config = "conf/model/object_detection/yolov8/yolov8s.toml"',
                    "gpu_mem_mib = 1800",
                    "gpu_mem_at_workers = 1",
                    'gpu_mem_source = "seed"',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        b.write_text(
            "\n".join(
                [
                    "[pack]",
                    "gpu_memory_budget_mib = 7000",
                    "[pack.libface]",
                    "worker_nums = 1",
                    "gpu_mem_mib = 1800",
                    "gpu_mem_at_workers = 1",
                    'gpu_mem_source = "seed"',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        out = root / "combined.toml"
        text = compose([a, b], out)
        if "[pack.yolov8s]" not in text or "[pack.libface]" not in text:
            print("self-test: missing composed sections", file=sys.stderr)
            return 1
        if text.index("[pack.yolov8s]") > text.index("[pack.libface]"):
            print("self-test: section order must follow arguments", file=sys.stderr)
            return 1
        if "gpu_memory_budget_mib = 7000" not in text or not text.startswith("# Composed"):
            print("self-test: global table or header missing", file=sys.stderr)
            return 1
        if "seed a" in text:
            print("self-test: source comments must not leak into the body", file=sys.stderr)
            return 1
        if "duplicate pack id libface" not in _expect_error([out, b]):
            print("self-test: duplicate id must fail", file=sys.stderr)
            return 1
        c = root / "c.toml"
        c.write_text(
            "[pack]\ngpu_memory_budget_mib = 1\n[pack.resnet]\nworker_nums = 1\n",
            encoding="utf-8",
        )
        conflict = _expect_error([b, c])
        if "gpu_memory_budget_mib" not in conflict or "conflicts" not in conflict:
            print("self-test: global key conflict must fail: %s" % conflict, file=sys.stderr)
            return 1
        missing = _expect_error([root / "nope.toml"])
        if "pack not found" not in missing:
            print("self-test: missing file must fail", file=sys.stderr)
            return 1
        bad = root / "bad.toml"
        bad.write_text("[other]\nworker_nums = 1\n", encoding="utf-8")
        if "unsupported table" not in _expect_error([bad]):
            print("self-test: unknown table must fail", file=sys.stderr)
            return 1
        empty = root / "empty.toml"
        empty.write_text("# only a comment\n", encoding="utf-8")
        if "no [pack.<id>]" not in _expect_error([empty]):
            print("self-test: pack without an id must fail", file=sys.stderr)
            return 1
        dup_key = root / "dup.toml"
        dup_key.write_text(
            "[pack.yolov8n]\nworker_nums = 1\nworker_nums = 2\n",
            encoding="utf-8",
        )
        if "duplicate key" not in _expect_error([dup_key]):
            print("self-test: duplicate key must fail", file=sys.stderr)
            return 1
        try:
            refuse_repo_output(ROOT / "conf" / "packs" / "composed.toml")
            print("self-test: conf/packs output must be refused", file=sys.stderr)
            return 1
        except ComposeError:
            pass
        try:
            refuse_repo_output(ROOT / "conf" / "server" / "composed.toml")
            print("self-test: conf/server output must be refused", file=sys.stderr)
            return 1
        except ComposeError:
            pass
        # Re-compose the two-id file alone: one id section each, global kept.
        solo = compose_text([out])
        if solo.count("[pack.yolov8s]") != 1 or solo.count("[pack.libface]") != 1:
            print("self-test: composed file must parse back to two ids", file=sys.stderr)
            return 1
    print("compose_packs.py self-test passed")
    return 0


def _expect_error(paths: list[Path]) -> str:
    try:
        compose_text(paths)
    except ComposeError as exc:
        return str(exc)
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "packs",
        nargs="*",
        type=Path,
        help="pack toml files, one or more [pack.<id>] sections each",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="machine-local combined pack (refuses conf/packs and conf/server)",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.packs or args.output is None:
        parser.error("pass pack files and -o/--output (or --self-test)")
    try:
        compose(args.packs, args.output)
    except ComposeError as exc:
        print("compose_packs: %s" % exc, file=sys.stderr)
        return 1
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
