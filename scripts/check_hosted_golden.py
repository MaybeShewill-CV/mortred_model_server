#!/usr/bin/env python3
"""Fail-closed contract for hosted CPU goldens, GPU smoke, and catalog CI tiers.

conf/ci_hosted_golden.json is the single source of truth. This script checks:

- each hosted gtest name exists in test/model_golden_test.cc
- hosted configs are MNN/ONNX (not TensorRT / .engine)
- listed weights exist in conf/weights_manifest.json, on_hf, profiles contain cpu
- hosted download size stays under max_bytes
- at least min_families distinct task families
- MORTRED_GPU_SMOKE_FILTER in .github/workflows/ci.yml matches gpu_smoke.cases
- every HTTP catalog id has a catalog_tiers entry (hosted | gpu-smoke | nightly)

Usage (repo root):
  python3 scripts/check_hosted_golden.py
  python3 scripts/check_hosted_golden.py --print-gtest-filter
  python3 scripts/check_hosted_golden.py --print-only-substrings
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from repo_toml import load_toml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "conf" / "ci_hosted_golden.json"
MANIFEST = ROOT / "conf" / "weights_manifest.json"
GOLDEN_CC = ROOT / "test" / "model_golden_test.cc"
CI_YML = ROOT / ".github" / "workflows" / "ci.yml"
ALLOWED_BACKENDS = {"mnn", "onnx"}
FORBIDDEN_SUFFIXES = (".engine",)
ALLOWED_TIERS = {"hosted", "gpu-smoke", "nightly"}
_GTEST_NAME_RE = re.compile(
    r"(?:TEST\s*\(\s*model_golden\s*,\s*([A-Za-z0-9_]+)\s*\)"
    r"|GOLDEN_[A-Z_]+_CASE\s*\(\s*([A-Za-z0-9_]+)\s*,)"
)
_SMOKE_FILTER_RE = re.compile(
    r"MORTRED_GPU_SMOKE_FILTER:\s*>-\s*\n\s*(model_golden[^\n]+)",
    re.MULTILINE,
)


def parse_cpp_http_models() -> set[str]:
    from check_consistency import parse_cpp_http_models as _parse

    return _parse()


def load_contract() -> dict:
    with CONTRACT.open(encoding="utf-8") as handle:
        return json.load(handle)


def gtest_names_in_source() -> set[str]:
    text = GOLDEN_CC.read_text(encoding="utf-8")
    names: set[str] = set()
    for match in _GTEST_NAME_RE.finditer(text):
        raw = match.group(1) or match.group(2)
        names.add(f"model_golden.{raw}")
    return names


def manifest_by_path() -> dict[str, dict]:
    with MANIFEST.open(encoding="utf-8-sig") as handle:
        data = json.load(handle)
    return {item["path"]: item for item in data.get("files", [])}


def backend_tables(node: object) -> list[dict]:
    tables: list[dict] = []
    if not isinstance(node, dict):
        return tables
    for key, value in node.items():
        if not isinstance(value, dict):
            continue
        if key == "backend" or str(key).endswith(".backend"):
            tables.append(value)
        else:
            tables.extend(backend_tables(value))
    return tables


def check_hosted_case(case: dict, known_gtest: set[str], by_path: dict[str, dict]) -> list[str]:
    errors: list[str] = []
    gtest = case.get("gtest")
    if not gtest:
        return ["hosted case missing gtest"]
    if gtest not in known_gtest:
        errors.append(f"{gtest}: not declared in test/model_golden_test.cc")
    config = case.get("config")
    if not config or not (ROOT / config).is_file():
        errors.append(f"{gtest}: config missing: {config}")
        return errors
    try:
        table = load_toml(ROOT / config)
    except (OSError, ValueError) as exc:
        errors.append(f"{gtest}: cannot parse {config}: {exc}")
        return errors
    backends = backend_tables(table)
    if not backends:
        errors.append(f"{gtest}: no [*.backend] table in {config}")
    listed = list(case.get("weights") or [])
    if not listed:
        errors.append(f"{gtest}: weights list is empty")
    listed_set = set(listed)
    for backend in backends:
        btype = str(backend.get("type") or "").lower()
        if btype and btype not in ALLOWED_BACKENDS:
            errors.append(
                f"{gtest}: backend.type={btype!r} is not CPU-runnable "
                f"(need one of {sorted(ALLOWED_BACKENDS)})"
            )
        model_path = backend.get("model_file_path") or ""
        rel = model_path[3:] if model_path.startswith("../") else model_path
        if any(rel.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            errors.append(f"{gtest}: model_file_path looks like a TRT engine: {rel}")
        if rel and rel not in listed_set:
            errors.append(
                f"{gtest}: {config} model_file_path {rel} is not in the hosted weights list"
            )
    for weight in listed:
        item = by_path.get(weight)
        if item is None:
            errors.append(f"{gtest}: {weight} is not in conf/weights_manifest.json")
            continue
        if item.get("on_hf") is not True:
            errors.append(f"{gtest}: {weight} is not on Hugging Face (on_hf != true)")
        profiles = item.get("profiles") or []
        if "cpu" not in profiles:
            errors.append(f"{gtest}: {weight} profiles must include cpu, got {profiles}")
    return errors


def check_ci_inference_contract() -> list[str]:
    errors: list[str] = []
    if not CONTRACT.is_file():
        return [f"missing {CONTRACT.relative_to(ROOT)}"]
    try:
        contract = load_contract()
    except json.JSONDecodeError as exc:
        return [f"{CONTRACT.relative_to(ROOT)}: invalid JSON: {exc}"]

    hosted = contract.get("hosted") or {}
    gpu_smoke = contract.get("gpu_smoke") or {}
    catalog_tiers = contract.get("catalog_tiers") or {}
    cases = hosted.get("cases") or []
    known_gtest = gtest_names_in_source()
    by_path = manifest_by_path()

    families: set[str] = set()
    hosted_ids: set[str] = set()
    total_bytes = 0
    seen_weights: set[str] = set()
    seen_gtest: set[str] = set()
    for case in cases:
        gtest = case.get("gtest")
        if gtest in seen_gtest:
            errors.append(f"duplicate hosted gtest: {gtest}")
        seen_gtest.add(gtest)
        errors.extend(check_hosted_case(case, known_gtest, by_path))
        family = case.get("family")
        if family:
            families.add(str(family))
        catalog_id = case.get("catalog_id")
        if catalog_id:
            hosted_ids.add(str(catalog_id))
        for weight in case.get("weights") or []:
            if weight in seen_weights:
                continue
            seen_weights.add(weight)
            item = by_path.get(weight) or {}
            total_bytes += int(item.get("size") or 0)

    min_families = int(hosted.get("min_families") or 4)
    if len(families) < min_families:
        errors.append(
            f"hosted golden covers {len(families)} families {sorted(families)}, "
            f"need at least {min_families}"
        )
    max_bytes = int(hosted.get("max_bytes") or 0)
    if max_bytes and total_bytes > max_bytes:
        errors.append(
            f"hosted weights total {total_bytes} bytes exceeds max_bytes={max_bytes}"
        )

    smoke_cases = gpu_smoke.get("cases") or []
    smoke_gtest: list[str] = []
    smoke_ids: set[str] = set()
    for case in smoke_cases:
        name = case.get("gtest")
        if not name:
            errors.append("gpu_smoke case missing gtest")
            continue
        if name not in known_gtest:
            errors.append(f"{name}: gpu smoke name not in test/model_golden_test.cc")
        smoke_gtest.append(name)
        catalog_id = case.get("catalog_id")
        if catalog_id:
            smoke_ids.add(str(catalog_id))

    yml = CI_YML.read_text(encoding="utf-8")
    match = _SMOKE_FILTER_RE.search(yml)
    if not match:
        errors.append("could not parse MORTRED_GPU_SMOKE_FILTER from .github/workflows/ci.yml")
    else:
        parsed = [part for part in match.group(1).strip().split(":") if part]
        if parsed != smoke_gtest:
            errors.append(
                "MORTRED_GPU_SMOKE_FILTER must match conf/ci_hosted_golden.json gpu_smoke.cases "
                f"order and names; yaml={parsed} json={smoke_gtest}"
            )

    http_ids = parse_cpp_http_models()
    if not http_ids:
        errors.append("failed to parse HTTP catalog ids from src/factory/*_task.h")
    extra = sorted(set(catalog_tiers) - http_ids)
    missing = sorted(http_ids - set(catalog_tiers))
    if extra:
        errors.append("catalog_tiers has unknown HTTP ids: " + ", ".join(extra))
    if missing:
        errors.append("HTTP catalog ids missing from catalog_tiers: " + ", ".join(missing))
    for model_id, tier in catalog_tiers.items():
        if tier not in ALLOWED_TIERS:
            errors.append(f"catalog_tiers[{model_id}]={tier!r} is not in {sorted(ALLOWED_TIERS)}")
        elif tier == "hosted" and model_id not in hosted_ids:
            errors.append(f"catalog_tiers[{model_id}]=hosted but no hosted case lists that catalog_id")
        elif tier == "gpu-smoke" and model_id not in smoke_ids:
            errors.append(
                f"catalog_tiers[{model_id}]=gpu-smoke but no gpu_smoke case lists that catalog_id"
            )

    return errors


def hosted_gtest_filter() -> str:
    contract = load_contract()
    return ":".join(case["gtest"] for case in contract["hosted"]["cases"])


def hosted_only_substrings() -> list[str]:
    contract = load_contract()
    seen: list[str] = []
    for case in contract["hosted"]["cases"]:
        for weight in case["weights"]:
            token = weight[len("weights/") :] if weight.startswith("weights/") else weight
            if token not in seen:
                seen.append(token)
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-gtest-filter",
        action="store_true",
        help="print the hosted --gtest_filter and exit (no checks)",
    )
    parser.add_argument(
        "--print-only-substrings",
        action="store_true",
        help="print fetch_weights.py --only tokens, one per line",
    )
    args = parser.parse_args()
    if args.print_gtest_filter:
        print(hosted_gtest_filter())
        return 0
    if args.print_only_substrings:
        for token in hosted_only_substrings():
            print(token)
        return 0

    errors = check_ci_inference_contract()
    if errors:
        print("Hosted golden / CI tier contract failed:")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("Hosted golden / CI tier contract passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
