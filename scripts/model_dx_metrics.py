#!/usr/bin/env python3
"""Collect reproducible model-layer developer-experience metrics.

The numbers are intentionally source-based rather than runtime-based: they
measure how much boilerplate a model author currently has to understand and
copy. Output is deterministic so two runs on the same commit produce identical
JSON/Markdown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


MODEL_EXTENSIONS = {".h", ".inl", ".cpp", ".cc"}


def source_files(root: Path, *directories: str, extensions: set[str] | None = None) -> List[Path]:
    extensions = extensions or MODEL_EXTENSIONS
    if not directories:
        return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix in extensions)
    result: List[Path] = []
    for directory in directories:
        base = root / directory
        result.extend(path for path in base.rglob("*") if path.is_file() and path.suffix in extensions)
    return sorted(result)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def count_pattern(paths: Iterable[Path], pattern: str) -> int:
    return sum(read(path).count(pattern) for path in paths)


def line_count(paths: Iterable[Path]) -> int:
    return sum(len(read(path).splitlines()) for path in paths)


def collect_metrics(repo_root: Path) -> Dict[str, object]:
    models_root = repo_root / "src" / "models"
    models = source_files(models_root)
    model_impl = [path for path in models if "backend" not in path.parts]
    factory = source_files(repo_root / "src" / "factory", extensions={".h", ".cpp"})
    tests = source_files(repo_root / "test")
    configs = sorted((repo_root / "conf" / "model").rglob("*.toml"))
    golden = sorted((repo_root / "test" / "golden").glob("*"))

    model_text = "\n".join(read(path) for path in model_impl)
    factory_text = "\n".join(read(path) for path in factory)
    test_text = "\n".join(read(path) for path in tests)

    public_model_names = sorted({path.parent.name for path in configs})

    def task_name(path: Path) -> str:
        relative = path.parent.relative_to(models_root)
        return relative.parts[0] if relative.parts else "(root)"

    task_counts = Counter(task_name(path) for path in model_impl)
    metrics: Dict[str, object] = {
        "source": {
            "model_files": len(models),
            "model_implementation_files": len(model_impl),
            "model_lines": line_count(models),
            "model_implementation_lines": line_count(model_impl),
            "factory_files": len(factory),
            "factory_lines": line_count(factory),
            "model_config_files": len(configs),
            "golden_files": len(golden),
        },
        "boilerplate": {
            "model_memcpy_calls": count_pattern(model_impl, "std::memcpy"),
            "model_chw_converter_calls": count_pattern(model_impl, "convert_to_chw_vec"),
            "model_make_f32_calls": count_pattern(model_impl, "Tensor::make<float>"),
            "model_resize_calls": count_pattern(model_impl, "cv::resize("),
            "model_cvt_color_calls": count_pattern(model_impl, "cv::cvtColor("),
        },
        "contracts": {
            "validated_f32_output_calls": count_pattern(model_impl, "validated_f32_"),
            "validate_output_tensor_calls": count_pattern(model_impl, "validate_output_tensor("),
            "request_geometry_calls": count_pattern(model_impl, "make_geometry_scale(")
            + count_pattern(model_impl, "validated_source_size("),
        },
        "hooks": {
            "preprocess_functions": model_text.count(" preprocess("),
            "postprocess_functions": model_text.count(" postprocess("),
            "prepare_inputs_functions": model_text.count(" prepare_inputs("),
            "run_sessions_functions": model_text.count(" run_sessions("),
        },
        "factory_server": {
            "create_model_functions": factory_text.count("create_"),
            "create_server_functions": factory_text.count("_server("),
            "server_spec_blocks": factory_text.count("CvServerSpec<"),
            "model_section_assignments": factory_text.count(".model_section ="),
            "server_section_assignments": factory_text.count(".server_section ="),
            "fill_response_assignments": factory_text.count(".fill_response ="),
        },
        "tests": {
            "contract_test_files": len([path for path in tests if "contract" in path.name]),
            "golden_case_registrations": test_text.count("TEST(model_golden,"),
            "model_contract_tests": test_text.count("TEST(ModelOutputContract,"),
            "object_contract_tests": test_text.count("TEST(ObjectDetectionOutputContract,"),
        },
        "tasks": dict(sorted(task_counts.items())),
        "public_model_names": public_model_names,
    }
    serialized = json.dumps(metrics, sort_keys=True, separators=(",", ":"))
    metrics["digest"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return metrics


def markdown(metrics: Dict[str, object]) -> str:
    source = metrics["source"]
    boilerplate = metrics["boilerplate"]
    contracts = metrics["contracts"]
    hooks = metrics["hooks"]
    factory_server = metrics["factory_server"]
    tests = metrics["tests"]
    tasks = metrics["tasks"]

    lines = [
        "# P4 Model Developer Experience Metrics",
        "",
        "This report is generated by `scripts/model_dx_metrics.py` and is deterministic for a given commit.",
        "",
        "## Source scale",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in source.items():  # type: ignore[union-attr]
        lines.append(f"| `{key}` | {value} |")

    lines.extend(["", "## Model author boilerplate", "", "| Metric | Value |", "|---|---:|"])
    for key, value in boilerplate.items():  # type: ignore[union-attr]
        lines.append(f"| `{key}` | {value} |")

    lines.extend(["", "## Contract usage", "", "| Metric | Value |", "|---|---:|"])
    for key, value in contracts.items():  # type: ignore[union-attr]
        lines.append(f"| `{key}` | {value} |")

    lines.extend(["", "## Model hooks", "", "| Metric | Value |", "|---|---:|"])
    for key, value in hooks.items():  # type: ignore[union-attr]
        lines.append(f"| `{key}` | {value} |")

    lines.extend(["", "## Factory / server registration", "", "| Metric | Value |", "|---|---:|"])
    for key, value in factory_server.items():  # type: ignore[union-attr]
        lines.append(f"| `{key}` | {value} |")

    lines.extend(["", "## Tests", "", "| Metric | Value |", "|---|---:|"])
    for key, value in tests.items():  # type: ignore[union-attr]
        lines.append(f"| `{key}` | {value} |")

    lines.extend(["", "## Model implementation files by task", "", "| Task | Files |", "|---|---:|"])
    for task, count in tasks.items():  # type: ignore[union-attr]
        lines.append(f"| `{task}` | {count} |")

    lines.extend(["", f"Metrics digest: `{metrics['digest']}`", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path, help="Write the report to this path instead of stdout")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = collect_metrics(args.root.resolve())
    if args.format == "json":
        content = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    else:
        content = markdown(metrics)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
