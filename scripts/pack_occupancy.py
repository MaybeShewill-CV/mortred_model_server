#!/usr/bin/env python3
"""Occupancy stamps for a machine pack.

TensorRT pack ids need gpu_mem_mib from calibrate --write-pack. occupancy_policy=off
or MORTRED_OCCUPANCY_ENFORCE=0 is unsafe: doctor --strict fails. Used by
mortredctl doctor. Does not spawn models.

Optional --measured-json (catalog id -> MiB) is the production entry for the
declared-vs-measured occupancy gate. Same tolerance and message as
occupancy_ready_for_spawn in src/control/occupancy_gate.h. Supervisor does not
sample model VRAM for this check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from pack_trt import pack_ids, pack_tensorrt_ids  # noqa: E402
from repo_toml import load_toml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _as_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().strip('"')


def _as_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = _as_str(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def pack_id_table(table: dict, catalog_id: str) -> dict:
    pack = table.get("pack")
    if isinstance(pack, dict) and catalog_id in pack and isinstance(pack[catalog_id], dict):
        return pack[catalog_id]
    kv = table.get("pack.%s" % catalog_id)
    return kv if isinstance(kv, dict) else {}


def occupancy_policy_of(table: dict) -> tuple[str, str]:
    """Return (normalized policy, raw) . Empty raw means omitted (enforce)."""
    pack = table.get("pack")
    raw = ""
    if isinstance(pack, dict):
        raw = _as_str(pack.get("occupancy_policy"))
    return (raw.lower() if raw else "enforce"), raw


def occupancy_disabled(policy: str) -> bool:
    env = os.environ.get("MORTRED_OCCUPANCY_ENFORCE", "").strip().lower()
    if env in ("0", "false", "off"):
        return True
    return policy == "off"


def calibrate_hint(pack: Path) -> str:
    return "stop the supervisor, then: mortredctl calibrate --pack %s --write-pack" % pack


def declared_measured_tol_mib(declared_mib: int) -> int:
    """Same rule as occupancy_declared_measured_tol_mib in occupancy_gate.h."""
    if declared_mib <= 0:
        return 64
    return max(64, declared_mib // 100)


def declared_measured_detail(catalog_id: str, declared_mib: int, measured_mib: int) -> str:
    """Same detail as occupancy_declared_measured_detail in occupancy_gate.h."""
    return "declared gpu_mem_mib=%d disagrees with measured %d for %s" % (
        declared_mib,
        measured_mib,
        catalog_id,
    )


def occupancy_gate_error(detail: str, pack: Path) -> str:
    return "occupancy gate: %s; %s" % (detail, calibrate_hint(pack))


def load_measured_json(path: Path) -> dict[str, int]:
    """Parse catalog id -> MiB map. Values must be integers (or whole floats)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("measured-json must be an object of catalog_id -> MiB")
    out: dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            raise ValueError("measured-json keys must be non-empty catalog ids")
        mib = _as_int(value)
        if mib is None:
            raise ValueError("measured-json[%s] must be an integer MiB" % key)
        out[key] = mib
    return out


def check_occupancy(
    pack: Path,
    project_root: Path,
    measured: dict[str, int] | None = None,
) -> list[str]:
    errors: list[str] = []
    try:
        table = load_toml(pack)
    except (OSError, ValueError) as exc:
        return ["cannot parse pack: %s" % exc]
    policy, raw = occupancy_policy_of(table)
    if raw and policy not in ("enforce", "off"):
        return [
            "invalid occupancy_policy='%s'; occupancy_policy must be enforce|off" % raw
        ]
    if occupancy_disabled(policy):
        return [
            "occupancy_policy=off (or MORTRED_OCCUPANCY_ENFORCE=0) is unsafe; "
            "set occupancy_policy=enforce, unset MORTRED_OCCUPANCY_ENFORCE, then: "
            "mortredctl calibrate --pack %s --write-pack" % pack
        ]
    measured = measured or {}
    for pid in pack_tensorrt_ids(pack, project_root):
        kv = pack_id_table(table, pid)
        mib = _as_int(kv.get("gpu_mem_mib"))
        if mib is None or mib <= 0:
            errors.append(
                "%s: missing gpu_mem_mib; %s" % (pid, calibrate_hint(pack))
            )
            continue
        workers = _as_int(kv.get("worker_nums"))
        at = _as_int(kv.get("gpu_mem_at_workers"))
        if workers is not None and at is not None and workers != at:
            errors.append(
                "%s: worker_nums=%d but gpu_mem_at_workers=%d; %s"
                % (pid, workers, at, calibrate_hint(pack))
            )
            continue
        if pid in measured:
            measured_mib = measured[pid]
            tol = declared_measured_tol_mib(mib)
            delta = abs(mib - measured_mib)
            if delta > tol:
                errors.append(
                    occupancy_gate_error(
                        declared_measured_detail(pid, mib, measured_mib), pack
                    )
                )
    return errors


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "conf" / "server" / "x").mkdir(parents=True)
        (root / "conf" / "model").mkdir(parents=True)
        (root / "conf" / "packs").mkdir(parents=True)
        (root / "_bin").mkdir()
        (root / "conf" / "server" / "x" / "s.toml").write_text(
            '[X_SERVER]\nmodel="X"\nprofile="any"\n'
            '[X]\nmodel_config_file_path="../conf/model/x.toml"\n',
            encoding="utf-8",
        )
        (root / "conf" / "model" / "x.toml").write_text(
            '[X.backend]\ntype="tensorrt"\nmodel_file_path="../weights/x.engine"\n',
            encoding="utf-8",
        )
        pack = root / "conf" / "packs" / "p.toml"
        pack.write_text("[pack.X]\nworker_nums=1\n", encoding="utf-8")
        errs = check_occupancy(pack, root)
        if not errs or "missing gpu_mem_mib" not in errs[0]:
            print("self-test: missing stamp must fail", errs, file=sys.stderr)
            return 1
        if "calibrate --pack" not in errs[0]:
            print("self-test: missing stamp must name calibrate", errs, file=sys.stderr)
            return 1
        pack.write_text(
            "[pack.X]\nworker_nums=1\ngpu_mem_mib=1800\ngpu_mem_at_workers=1\n",
            encoding="utf-8",
        )
        if check_occupancy(pack, root):
            print("self-test: stamped TRT must pass", file=sys.stderr)
            return 1
        # No measured map: declared/measured gate must not fire.
        if check_occupancy(pack, root, measured=None):
            print("self-test: no measured must pass", file=sys.stderr)
            return 1
        within = check_occupancy(pack, root, measured={"X": 1864})
        if within:
            print("self-test: within tol must pass", within, file=sys.stderr)
            return 1
        over = check_occupancy(pack, root, measured={"X": 2000})
        if not over:
            print("self-test: over tol must fail", file=sys.stderr)
            return 1
        msg = over[0]
        if not msg.startswith("occupancy gate:"):
            print("self-test: over tol prefix", msg, file=sys.stderr)
            return 1
        if "1800" not in msg or "2000" not in msg:
            print("self-test: over tol must name declared and measured", msg, file=sys.stderr)
            return 1
        if "mortredctl calibrate" not in msg or "--write-pack" not in msg:
            print("self-test: over tol must name calibrate --write-pack", msg, file=sys.stderr)
            return 1
        pack.write_text(
            "[pack.X]\nworker_nums=4\ngpu_mem_mib=1800\ngpu_mem_at_workers=1\n",
            encoding="utf-8",
        )
        stale = check_occupancy(pack, root)
        if not stale or "gpu_mem_at_workers" not in stale[0]:
            print("self-test: stale w* must fail", stale, file=sys.stderr)
            return 1
        pack.write_text(
            '[pack]\noccupancy_policy="off"\n[pack.X]\nworker_nums=1\n',
            encoding="utf-8",
        )
        off = check_occupancy(pack, root)
        if not off or "unsafe" not in off[0]:
            print("self-test: policy=off must fail doctor check", off, file=sys.stderr)
            return 1
        (root / "conf" / "model" / "x.toml").write_text(
            '[X.backend]\ntype="onnx"\nmodel_file_path="../weights/x.onnx"\n',
            encoding="utf-8",
        )
        pack.write_text("[pack.X]\nworker_nums=1\n", encoding="utf-8")
        if check_occupancy(pack, root):
            print("self-test: non-TRT without stamp must pass", file=sys.stderr)
            return 1
    demo = ROOT / "conf" / "packs" / "demo.toml"
    if demo.is_file() and check_occupancy(demo, ROOT):
        print("self-test: demo pack must not require occupancy stamps", file=sys.stderr)
        return 1
    print("pack_occupancy.py self-test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, help="pack toml")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--measured-json",
        type=Path,
        help="optional catalog id -> measured MiB JSON for declared/measured gate",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.pack is None:
        parser.error("--pack is required unless --self-test")
    pack = args.pack if args.pack.is_absolute() else args.project_root / args.pack
    if not pack.is_file():
        print("pack not found: %s" % pack, file=sys.stderr)
        return 1
    print("pack: %s" % pack)
    ids = pack_ids(pack)
    print("ids: %s" % (" ".join(ids) if ids else "(none)"))
    if not args.check:
        parser.error("pass --check or --self-test")
        return 2
    measured: dict[str, int] | None = None
    if args.measured_json is not None:
        mj = (
            args.measured_json
            if args.measured_json.is_absolute()
            else args.project_root / args.measured_json
        )
        try:
            measured = load_measured_json(mj)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print("measured-json: %s" % exc, file=sys.stderr)
            return 1
    errors = check_occupancy(pack, args.project_root, measured=measured)
    for err in errors:
        print(err, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
