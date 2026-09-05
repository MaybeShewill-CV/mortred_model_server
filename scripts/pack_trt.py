#!/usr/bin/env python3
"""Resolve TensorRT engines for a machine pack.

Walks [pack.<ID>] ids, the matching conf/server model_config (or pack
model_config override), and every *backend table with type=tensorrt.
Used by prepare_pack.sh and mortredctl doctor. Does not run trtexec.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from repo_toml import load_toml

ROOT = Path(__file__).resolve().parents[1]


def _as_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().strip('"')


def _walk_tables(obj: object, prefix: str = "") -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    if not isinstance(obj, dict):
        return rows
    rows.append((prefix, obj))
    for key, val in obj.items():
        if isinstance(val, dict):
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_walk_tables(val, child))
    return rows


def pack_ids(pack_path: Path) -> list[str]:
    ids: list[str] = []
    for line in pack_path.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if t.startswith("[pack.") and t.endswith("]"):
            pid = t[6:-1].strip()
            if pid:
                ids.append(pid)
    return ids


def pack_model_config(pack_path: Path, catalog_id: str, project_root: Path) -> Path | None:
    table = load_toml(pack_path)
    # tomllib nests pack.ID; fallback parser uses key "pack.ID"
    kv = None
    if isinstance(table.get("pack"), dict) and catalog_id in table["pack"]:
        kv = table["pack"][catalog_id]
    if kv is None:
        kv = table.get(f"pack.{catalog_id}")
    if not isinstance(kv, dict):
        return None
    rel = _as_str(kv.get("model_config"))
    if not rel:
        return None
    path = Path(rel)
    if not path.is_absolute():
        path = project_root / path
    return path


def find_server_toml(project_root: Path, catalog_id: str) -> Path | None:
    conf_server = project_root / "conf" / "server"
    if not conf_server.is_dir():
        return None
    for cfg in sorted(conf_server.rglob("*.toml")):
        try:
            table = load_toml(cfg)
        except (OSError, ValueError):
            continue
        for kv in table.values():
            if isinstance(kv, dict) and _as_str(kv.get("model")) == catalog_id:
                return cfg
    return None


def server_model_config(server_toml: Path, project_root: Path) -> Path | None:
    try:
        table = load_toml(server_toml)
    except (OSError, ValueError):
        return None
    for name, kv in table.items():
        if not isinstance(kv, dict) or str(name).endswith("_SERVER"):
            continue
        rel = _as_str(kv.get("model_config_file_path"))
        if not rel:
            continue
        path = Path(rel)
        if not path.is_absolute():
            path = (project_root / "_bin" / path).resolve()
            if not path.is_file():
                path = (project_root / rel).resolve()
        return path
    return None


def trt_engine_paths(model_toml: Path, project_root: Path) -> list[Path]:
    try:
        table = load_toml(model_toml)
    except (OSError, ValueError):
        return []
    engines: list[Path] = []
    for name, kv in _walk_tables(table):
        if "backend" not in name.lower():
            continue
        if _as_str(kv.get("type")).lower() != "tensorrt":
            continue
        raw = _as_str(kv.get("model_file_path"))
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            from_bin = (project_root / "_bin" / path).resolve()
            from_root = (project_root / path).resolve()
            path = from_bin if from_bin.is_file() else from_root
            if not path.is_file():
                path = from_bin
        engines.append(path)
    return engines


def pack_trt_engines(pack_path: Path, project_root: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for pid in pack_ids(pack_path):
        override = pack_model_config(pack_path, pid, project_root)
        model_toml = override
        if model_toml is None:
            server = find_server_toml(project_root, pid)
            if server is None:
                continue
            model_toml = server_model_config(server, project_root)
        if model_toml is None or not model_toml.is_file():
            continue
        for engine in trt_engine_paths(model_toml, project_root):
            rows.append((pid, engine))
    return rows


def convert_filters(engines: list[Path], project_root: Path, manifest: Path) -> list[str]:
    declared = []
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        declared = [e.get("engine", "") for e in data.get("engines", [])]
    filters: list[str] = []
    for engine in engines:
        try:
            rel = engine.resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError:
            rel = engine.name
        hit = next((d for d in declared if d and (d in rel or Path(d).name == engine.name)), "")
        token = Path(hit).name if hit else engine.name
        if token and token not in filters:
            filters.append(token)
    return filters


def check_engines(rows: list[tuple[str, Path]]) -> list[str]:
    errors: list[str] = []
    for pid, engine in rows:
        if not engine.is_file() or engine.stat().st_size == 0:
            errors.append(f"{pid}: missing or empty {engine}")
    return errors


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "conf" / "server" / "x").mkdir(parents=True)
        (root / "conf" / "model").mkdir(parents=True)
        (root / "conf" / "packs").mkdir(parents=True)
        (root / "_bin").mkdir()
        (root / "weights").mkdir()
        (root / "conf" / "server" / "x" / "s.toml").write_text(
            '[X_SERVER]\nmodel="X"\n[X]\nmodel_config_file_path="../conf/model/x.toml"\n',
            encoding="utf-8",
        )
        (root / "conf" / "model" / "x.toml").write_text(
            '[X.backend]\ntype="tensorrt"\nmodel_file_path="../weights/x.engine"\n',
            encoding="utf-8",
        )
        (root / "conf" / "packs" / "p.toml").write_text("[pack.X]\nworker_nums=1\n", encoding="utf-8")
        rows = pack_trt_engines(root / "conf" / "packs" / "p.toml", root)
        if len(rows) != 1 or rows[0][0] != "X":
            print("self-test: expected one TRT row", file=sys.stderr)
            return 1
        if not check_engines(rows):
            print("self-test: missing engine must fail check", file=sys.stderr)
            return 1
        engine = root / "weights" / "x.engine"
        engine.write_bytes(b"not-empty")
        rows = pack_trt_engines(root / "conf" / "packs" / "p.toml", root)
        if check_engines(rows):
            print("self-test: present engine must pass", file=sys.stderr)
            return 1
        mnn = root / "conf" / "model" / "x.toml"
        mnn.write_text('[X.backend]\ntype="mnn"\nmodel_file_path="../weights/x.mnn"\n', encoding="utf-8")
        if pack_trt_engines(root / "conf" / "packs" / "p.toml", root):
            print("self-test: mnn backend must not list engines", file=sys.stderr)
            return 1
    print("pack_trt.py self-test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, help="pack toml")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--convert-filters", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.pack is None:
        parser.error("--pack is required unless --self-test")
    pack = args.pack if args.pack.is_absolute() else args.project_root / args.pack
    if not pack.is_file():
        print(f"pack not found: {pack}", file=sys.stderr)
        return 1
    rows = pack_trt_engines(pack, args.project_root)
    if args.convert_filters:
        for token in convert_filters([e for _, e in rows], args.project_root,
                                     args.project_root / "conf" / "trt_engines.json"):
            print(token)
        return 0
    if args.list:
        if not rows:
            print("# no TensorRT backends in pack")
            return 0
        for pid, engine in rows:
            print(f"{pid}\t{engine}")
        return 0
    if args.check:
        errors = check_engines(rows)
        for err in errors:
            print(err, file=sys.stderr)
        return 1 if errors else 0
    parser.error("pass --list, --check, --convert-filters, or --self-test")
    return 2


if __name__ == "__main__":
    sys.exit(main())
