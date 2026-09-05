#!/usr/bin/env python3
"""Generate the skeleton of a new model.

The generated files are deliberately small and always compilable: the three
model hooks return an explicit MODEL_NOT_IMPLEMENTED status instead of
pretending to work, so a half-finished model can never be served by accident.

    python scripts/new_model.py --list-tasks
    python scripts/new_model.py --task object_detection --name rtdetr \\
        --class RtdetrDetector --dry-run

The scaffolder never edits existing files unless `--apply-catalog` is passed.
That flag inserts the catalog `Entry{...}` into the matching `*_task.h`.
Creator functions and test CMake registration are still printed as snippets.
A new catalog row is enough for `mortred-model-server.out --model SECTION` and
`mortred-model-benchmark.out --model SECTION`; no new ELF or CMake app target.

Generated C++ is passed through clang-format when the tool is available, so the
templates can stay readable instead of hand-tuning line breaks that only work
for one class name length.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repo_toml import load_toml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = REPOSITORY_ROOT
TEMPLATE_DIR = REPO_ROOT / "templates" / "model"

PLACEHOLDER = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
UPPER_IDENTIFIER = re.compile(r"^[A-Z][A-Z0-9_]*$")

BACKENDS = {
    "mnn": {"device": "gpu", "weight_ext": "mnn", "extra": 'input_layout = "nhwc"\nthreads = 4'},
    "onnx": {"device": "gpu", "weight_ext": "onnx", "extra": ""},
    "tensorrt": {"device": "gpu", "weight_ext": "engine", "extra": ""},
}


class ScaffoldError(Exception):
    """User facing error: bad arguments or an existing file in the way."""


def load_tasks() -> dict:
    with (TEMPLATE_DIR / "tasks.json").open(encoding="utf-8") as handle:
        return json.load(handle)["tasks"]


def snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def upper_snake(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()


def render(template: str, values: dict) -> str:
    def substitute(match: re.Match) -> str:
        key = match.group(1)
        if key not in values:
            raise ScaffoldError(f"template placeholder {{{{key}}}} has no value")
        return str(values[key])

    return PLACEHOLDER.sub(substitute, template)


def build_values(task: str, spec: dict, name: str, class_name: str, backend: str) -> dict:
    backend_spec = BACKENDS[backend]
    file_base = snake_case(class_name)
    section = upper_snake(name)
    return {
        "TASK": task,
        "NAME": name,
        "CLASS_NAME": class_name,
        "FILE_BASE": file_base,
        "SECTION": section,
        "GUARD": upper_snake(file_base),
        "MODEL_DIR": spec["model_dir"],
        "IO_NAMESPACE": spec["io_namespace"],
        "IO_HEADER": spec["io_header"],
        "OUTPUT_TYPE": spec["output_type"],
        "CATALOG_HEADER": spec["catalog_header"],
        "BACKEND": backend,
        "DEVICE": backend_spec["device"],
        "WEIGHT_EXT": backend_spec["weight_ext"],
        "BACKEND_EXTRA": backend_spec["extra"],
        "DATE": __import__("datetime").date.today().isoformat(),
        "SERVER_SECTION_SUFFIX": spec.get("server_section_suffix", ""),
        "PORT": spec.get("_port", ""),
        "SERVER_URI": spec.get("_uri", ""),
    }


def file_plan(values: dict) -> list[tuple[Path, str]]:
    """(destination, template file name) pairs, in the order they are written."""
    name = values["NAME"]
    model_dir = values["MODEL_DIR"]
    file_base = values["FILE_BASE"]
    plan = [
        (REPO_ROOT / "src" / "models" / model_dir / f"{file_base}.h", "model_header.h.in"),
        (REPO_ROOT / "src" / "models" / model_dir / f"{file_base}.inl", "model_impl.inl.in"),
        (REPO_ROOT / "conf" / "model" / model_dir / name / f"{name}_config.toml", "model_config.toml.in"),
        (REPO_ROOT / "test" / f"{file_base}_output_contract_unittest.cc", "output_contract_unittest.cc.in"),
        (REPO_ROOT / "docs" / "models" / model_dir / f"{name}.md", "model_readme.md.in"),
    ]
    if values.get("SERVER_SECTION_SUFFIX"):
        plan.append(
            (
                REPO_ROOT / "conf" / "server" / values["TASK"] / name / f"{name}_server_config.toml",
                "server_config.toml.in",
            )
        )
    return plan


def allocate_port_and_uri(task: str, name: str) -> tuple[int, str]:
    ports: set[int] = set()
    uris: set[str] = set()
    conf_server = REPO_ROOT / "conf" / "server"
    if conf_server.exists():
        for cfg in conf_server.rglob("*.toml"):
            try:
                table = load_toml(cfg)
            except (ValueError, OSError):
                continue
            for sec, kv in table.items():
                if not str(sec).endswith("_SERVER") or not isinstance(kv, dict):
                    continue
                if "port" in kv:
                    try:
                        ports.add(int(kv["port"]))
                    except (TypeError, ValueError):
                        pass
                if kv.get("server_uri"):
                    uris.add(str(kv["server_uri"]))
    port = 9100
    while port in ports:
        port += 1
    uri = f"/mortred_ai_server_v1/{task}/{name}"
    suffix = 2
    while uri in uris:
        uri = f"/mortred_ai_server_v1/{task}/{name}_{suffix}"
        suffix += 1
    return port, uri


def catalog_entry_line(task: str, spec: dict, values: dict) -> str:
    creator = "create_" + values["NAME"] + "_model"
    section = values["SECTION"]
    kind = "ObjectEntry" if task == "object_detection" else "Entry"
    filler = spec.get("response_filler")
    suffix = spec.get("server_section_suffix")
    if not (filler and suffix):
        return f'{kind}{{"{section}", "{values["CLASS_NAME"]}", &{creator}<Input, Output>}},'
    return (
        f'{kind}{{"{section}", "{values["CLASS_NAME"]}", "{section}{suffix}", '
        f'&{creator}<ImageInput, Output>, &jinq::server::response::{filler}}},'
    )


def apply_catalog_entry(task: str, spec: dict, values: dict) -> None:
    header = REPO_ROOT / "src" / spec["catalog_header"]
    if not header.is_file():
        raise ScaffoldError(f"catalog header missing: {header}")
    text = header.read_text(encoding="utf-8")
    fn = spec["catalog_function"]
    start = text.find(f"{fn}()")
    if start < 0:
        raise ScaffoldError(f"{header}: {fn}() not found")
    next_fn = text.find("\ninline ", start + 1)
    block_end = next_fn if next_fn > 0 else text.find("\n} //", start)
    if block_end < 0:
        block_end = len(text)
    block = text[start:block_end]
    close = block.rfind("    };")
    if close < 0:
        raise ScaffoldError(f"{header}: could not find catalog entries closing")
    entry = "        " + catalog_entry_line(task, spec, values) + "\n"
    header.write_text(text[:start] + block[:close] + entry + block[close:] + text[block_end:], encoding="utf-8")


def catalog_snippet(task: str, spec: dict, values: dict) -> str:
    """Registration snippet printed for the developer; never applied directly."""
    creator = "create_" + values["NAME"] + "_model"
    model_class = values["CLASS_NAME"]
    section = values["SECTION"]
    header = "src/factory/" + spec["catalog_header"].split("/", 1)[-1]

    creator_block = (
        "// " + header + "\n"
        "template <typename INPUT, typename OUTPUT>\n"
        "std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> " + creator + "(const std::string &model_name) {\n"
        "    (void)model_name;\n"
        "    return std::make_unique<" + model_class + "<INPUT, OUTPUT>>();\n"
        "}\n"
    )

    filler = spec.get("response_filler")
    suffix = spec.get("server_section_suffix")
    if not (filler and suffix):
        # model-only catalog: no server section, no response filler
        return (
            creator_block
            + "\nEntry{\"" + section + "\", \"" + model_class + "\",\n"
            "      &" + creator + "<Input, Output>},"
        )

    return (
        creator_block
        + "\nEntry{\"" + section + "\", \"" + model_class + "\", \"" + section + suffix + "\",\n"
        "      &" + creator + "<ImageInput, Output>,\n"
        "      &jinq::server::response::" + filler + "},"
    )


def cmake_snippet(values: dict) -> str:
    target = f"{values['FILE_BASE']}_output_contract_unittest"
    return (
        f"# test/CMakeLists.txt\n"
        f"list(APPEND TEST_LIST {target})\n\n"
        f"# ...inside the per-target elseif chain:\n"
        f'elseif(${{src}} STREQUAL "{target}")\n'
        f"    set(EXTRA_LIBS models glog::glog ${{OpenCV_LIBS}} vendored::mnn)"
    )


def golden_snippet(values: dict) -> str:
    return (
        f"# 1. put real weights under weights/{values['MODEL_DIR']}/{values['NAME']}/\n"
        f"# 2. add a golden baseline at test/golden/{values['NAME']}.json\n"
        f"# 3. register a case in test/model_golden_test.cc\n"
        f"# 4. only then replace the failing scaffold expectation in\n"
        f"#    test/{values['FILE_BASE']}_output_contract_unittest.cc"
    )


def generate(task: str, name: str, class_name: str, backend: str, dry_run: bool, force: bool,
             apply_catalog: bool = False) -> int:
    spec = load_tasks()[task]
    values = build_values(task, spec, name, class_name, backend)
    if values.get("SERVER_SECTION_SUFFIX"):
        port, uri = allocate_port_and_uri(task, name)
        values["PORT"] = str(port)
        values["SERVER_URI"] = uri
    plan = file_plan(values)

    conflicts = [path for path, _ in plan if path.exists()]
    if conflicts and not force:
        raise ScaffoldError(
            "refusing to overwrite existing file(s); pass --force to replace scaffolds you own:\n  "
            + "\n  ".join(str(path.relative_to(REPO_ROOT)) for path in conflicts)
        )

    print(f"task        : {task}")
    print(f"section     : [{values['SECTION']}]")
    print(f"class       : {values['CLASS_NAME']}")
    print(f"backend     : {backend} ({values['DEVICE']})")
    print("files       :")
    generated_sources = []
    for path, template_name in plan:
        flag = "overwrite" if path.exists() else "create"
        print(f"  [{flag}] {path.relative_to(REPO_ROOT)}  <-  templates/model/{template_name}")
        if dry_run:
            continue
        template = (TEMPLATE_DIR / template_name).read_text(encoding="utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(template, values), encoding="utf-8", newline="\n")
        if path.suffix in (".h", ".inl", ".cc"):
            generated_sources.append(path)

    if not dry_run:
        format_generated(generated_sources)

    if dry_run:
        print("\n(dry run: nothing written)")
        return 0

    print("\nmanual steps - the scaffolder intentionally does not edit shared files:\n")
    print("1) catalog entry\n" + catalog_snippet(task, spec, values) + "\n")
    if apply_catalog and not dry_run:
        apply_catalog_entry(task, spec, values)
        print(f"   applied catalog Entry into src/{spec['catalog_header']}\n")
    print("2) test registration\n" + cmake_snippet(values) + "\n")
    print("3) golden case\n" + golden_snippet(values) + "\n")
    print(f"4) verify\n  cmake --build <build-dir> --target {values['FILE_BASE']}_output_contract_unittest")
    print("\nAfter the catalog row exists, both unified entrypoints pick the model up:")
    print(f"  mortred-model-server.out --model {values['SECTION']} conf/server/...")
    print(f"  mortred-model-benchmark.out --model {values['SECTION']} conf/model/...")
    return 0


def format_generated(sources: list[Path]) -> None:
    """Normalise generated C++ with the repository clang-format config."""
    clang_format = shutil.which("clang-format")
    if clang_format is None:
        print("note: clang-format not found, run it on the generated files before committing")
        return
    result = subprocess.run(
        # resolve the config from the real repository: REPO_ROOT is redirected
        # into a temp tree while the self-test runs
        [clang_format, "-i", f"--style=file:{REPOSITORY_ROOT / '.clang-format'}",
         *(str(path) for path in sources)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ScaffoldError(f"clang-format failed on generated code:\n{(result.stdout + result.stderr).strip()}")


def self_test() -> int:
    """Exercise the scaffolder end to end inside a throwaway directory."""
    import contextlib

    tasks = load_tasks()
    if not tasks:
        raise ScaffoldError("templates/model/tasks.json declares no tasks")

    # 1. every template renders for every task x backend without leftovers
    for task, spec in sorted(tasks.items()):
        for backend in sorted(BACKENDS):
            values = build_values(task, spec, f"probe{len(task)}", f"Probe{task.title()}", backend)
            for _, template_name in file_plan(values):
                template = (TEMPLATE_DIR / template_name).read_text(encoding="utf-8")
                leftover = PLACEHOLDER.search(render(template, values))
                if leftover:
                    raise ScaffoldError(f"{task}/{backend}/{template_name}: unreplaced {leftover.group(0)}")

    # 2. dry-run writes nothing, a real run writes, no --force refuses to
    #    overwrite, and --force replaces. Everything happens under a temp root.
    task = "classification"
    backend = tasks[task]["default_backend"]

    def call(dry_run: bool, force: bool) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            generate(task, "probe", "ProbeModel", backend, dry_run, force, apply_catalog=False)

    with tempfile.TemporaryDirectory() as tmp:
        original_root = REPO_ROOT
        globals()["REPO_ROOT"] = Path(tmp)
        try:
            values = build_values(task, tasks[task], "probe", "ProbeModel", backend)
            written = [path for path, _ in file_plan(values)]

            call(dry_run=True, force=False)
            if any(path.exists() for path in written):
                raise ScaffoldError("--dry-run wrote files")

            call(dry_run=False, force=False)
            if not all(path.is_file() for path in written):
                raise ScaffoldError("generation did not create the expected files")

            for path in written:
                path.write_text("sentinel", encoding="utf-8")
            try:
                call(dry_run=False, force=False)
            except ScaffoldError:
                pass
            else:
                raise ScaffoldError("generation overwrote an existing file without --force")
            if any(path.read_text(encoding="utf-8") != "sentinel" for path in written):
                raise ScaffoldError("generation mutated files without --force")

            call(dry_run=False, force=True)
            if any(path.read_text(encoding="utf-8") == "sentinel" for path in written):
                raise ScaffoldError("--force did not replace the existing scaffold")

            clang_format = shutil.which("clang-format")
            if clang_format:
                sources = [str(path) for path in written if path.suffix in (".h", ".inl", ".cc")]
                # the temp tree has no .clang-format of its own, so point at the repo one
                style = f"--style=file:{REPOSITORY_ROOT / '.clang-format'}"
                result = subprocess.run(
                    [clang_format, "--dry-run", "--Werror", style, *sources],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    detail = (result.stdout + result.stderr).strip()
                    raise ScaffoldError(f"generated code is not clang-format clean:\n{detail}")
        finally:
            globals()["REPO_ROOT"] = original_root

    print(f"scaffolder self-test passed ({len(tasks)} tasks x {len(BACKENDS)} backends)")
    if not shutil.which("clang-format"):
        print("note: clang-format not found, generated-code formatting was not verified")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", help="task directory / io namespace, see --list-tasks")
    parser.add_argument("--name", help="model key, e.g. rtdetr -> [RTDETR] and rtdetr_config.toml")
    parser.add_argument("--class", dest="class_name", help="C++ class name, e.g. RtdetrDetector")
    parser.add_argument("--backend", choices=sorted(BACKENDS), help="inference backend (default: task default)")
    parser.add_argument("--dry-run", action="store_true", help="print the file plan without writing anything")
    parser.add_argument("--force", action="store_true", help="overwrite files that already exist")
    parser.add_argument("--apply-catalog", action="store_true",
                        help="insert the catalog Entry into the matching *_task.h")
    parser.add_argument("--list-tasks", action="store_true", help="print the supported tasks and exit")
    parser.add_argument("--check", action="store_true", help="run the scaffolder self-test and exit")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    tasks = load_tasks()

    if args.list_tasks:
        print("supported tasks (from templates/model/tasks.json):")
        for task, spec in sorted(tasks.items()):
            served = "served" if spec.get("response_filler") else "model-only"
            print(f"  {task:<22} {served:<11} output={spec['output_type']}")
        print("\ndiffusion samplers are intentionally not scaffolded: they have no image preprocess path.")
        return 0

    if args.check:
        return self_test()

    missing = [name for name, flag in (("task", args.task), ("name", args.name), ("class", args.class_name)) if not flag]
    if missing:
        raise ScaffoldError(f"missing required argument(s): {', '.join(missing)} (see --list-tasks)")
    if args.task not in tasks:
        raise ScaffoldError(f"unknown task '{args.task}' (see --list-tasks)")
    if not IDENTIFIER.match(args.class_name or ""):
        raise ScaffoldError(f"--class must be a CamelCase C++ identifier, got '{args.class_name}'")
    if not re.match(r"^[a-z][a-z0-9_]*$", args.name or ""):
        raise ScaffoldError(f"--name must be a lower_snake identifier, got '{args.name}'")

    backend = args.backend or tasks[args.task]["default_backend"]
    return generate(args.task, args.name, args.class_name, backend, args.dry_run, args.force,
                    apply_catalog=args.apply_catalog)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except ScaffoldError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
