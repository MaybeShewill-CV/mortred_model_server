#!/usr/bin/env python3
"""Repository consistency checker.

Verifies a few high-signal invariants:

1. Every source path referenced in docs/repository-layout.md exists.
2. `docs/repository-layout.md` paths exist (unified `mortred-model-server.out` /
   `mortred-model-benchmark.out` plus the control-plane binaries).
3. If _bin exists, stale binaries listed in the layout policy are reported.
4. Every conf/server category directory contains at least one TOML config.
5. Every conf/server/*.toml `model_config_file_path` points to an existing
   `.toml` file (the repo migrated model configs from .ini to .toml).
6. Every conf/server/*.toml `server_uri` is covered by docs/openapi.json paths.
7. The demo client (`scripts/server/test_server.py`) is self-contained:
   - orphaned config files (`config_utils.py`, `conf/py_demo/`) must not exist;
   - the client and `http_infer_rps.py` compile;
   - `test_server.py --list` and `http_infer_rps.py --self-test` exit 0 without
     locust / requests.
8. Every conf/server/*.toml follows the canonical two-section layout
   (`model_config_file_path` lives in the [MODEL] subtable, not the server
   section).
9. Every `*_SERVER` section declares `model=` matching the non-`_SERVER` table,
   `server_exe` is `mortred-model-server.out`, and the `model` set matches the
   HTTP subset of the C++ factory catalogs.
10. Every engine referenced by conf/model [*_TRT] sections is declared in
    conf/trt_engines.json, so the engine-regeneration script can never miss a
    config-required engine.
11. templates/model/tasks.json stays in sync with the real sources, so the
    scaffolder can never drift away from the C++ catalogs (catalog header,
    response filler, io namespace, output type, model directory).
12. src/models/**/*.inl uses exactly one TODO marker, TODO(new_model), so a
    scaffold is always greppable and half-finished models are easy to audit.
13. src/models/model_io_define.h stays a pure aggregate of src/models/io/*.h,
    and the IO headers include opencv2/core.hpp instead of the opencv.hpp
    umbrella header.
14. conf/ci_hosted_golden.json matches golden sources, HF cpu weights, the
    GPU smoke filter in ci.yml, and every HTTP catalog id has a CI tier.
15. `conf/packs/demo.toml` `[pack.<ID>]` ids exist as `model=` in conf/server.

Exit code 0 means consistent; non-zero means the repository needs attention.
"""

from __future__ import annotations

import argparse
import json
import py_compile
import re
import subprocess
import sys
from pathlib import Path

from repo_toml import load_toml

ROOT = Path(__file__).resolve().parents[1]

STALE_BINARIES = [
    "llama3_chatbot_server.out",
    "qwen2_vl_chatbot_server.out",
    "ollama_to_llama_cpp_proxy_server.out",
    "jina_embedding_v3_benchmark.out",
    "build_wiki_corpus_index.out",
    "search_wiki_corpus.out",
    "tokenizer_benchmark.out",
    "llm_request_parser_unittest",
    "llm_datatype_unittest",
    # the in-house ONNX->TRT converter was removed; external trtexec is used
    # instead (scripts/convert_trt_engines.sh)
    "onnx2trt_converter.out",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-stale-binaries",
        action="store_true",
        help="Also check _bin for stale executables without source.",
    )
    return parser.parse_args()


def check_layout_references() -> list[str]:
    errors: list[str] = []
    layout = ROOT / "docs" / "repository-layout.md"
    if not layout.exists():
        errors.append("docs/repository-layout.md is missing")
        return errors

    for line in layout.read_text(encoding="utf-8").splitlines():
        # Match inline code paths like `src/...` in the markdown tables/text.
        for match in re.finditer(r"`((?:src|conf|scripts|test|docs)/[^`]+)`", line):
            ref = match.group(1)
            # Strip trailing punctuation that may have been included in inline code.
            ref = ref.rstrip(".,;")
            # Ignore glob-like or placeholder references.
            if any(ch in ref for ch in "*?<>"):
                continue
            if not (ROOT / ref).exists():
                errors.append(f"docs/repository-layout.md references missing path: {ref}")
    return errors


def check_server_source_mapping() -> list[str]:
    """conf/server categories must contain TOML configs (the unified server
    binary covers every HTTP catalog id; there are no per-model sources)."""
    errors: list[str] = []
    conf_server = ROOT / "conf" / "server"
    if not conf_server.exists():
        errors.append("conf/server is missing")
        return errors
    for child in sorted(conf_server.iterdir()):
        if child.is_dir() and not list(child.rglob("*.toml")):
            errors.append(f"conf/server/{child.name} has no TOML configs")
    return errors


def check_stale_binaries() -> list[str]:
    errors: list[str] = []
    bin_dir = ROOT / "_bin"
    if not bin_dir.is_dir():
        return errors
    for name in STALE_BINARIES:
        if (bin_dir / name).exists():
            errors.append(f"stale binary without source: _bin/{name}")
    return errors


def check_server_model_config_paths() -> list[str]:
    """Every model_config_file_path must exist and use the .toml extension."""
    errors: list[str] = []
    conf_server = ROOT / "conf" / "server"
    if not conf_server.exists():
        return errors
    for cfg in sorted(conf_server.rglob("*.toml")):
        try:
            table = load_toml(cfg)
        except (ValueError, OSError) as exc:
            errors.append(f"conf/server config is not valid TOML: {cfg.relative_to(ROOT)} ({exc})")
            continue
        model_cfg_path = None
        for section in table.values():
            if isinstance(section, dict) and "model_config_file_path" in section:
                model_cfg_path = section["model_config_file_path"]
                break
        if model_cfg_path is None:
            errors.append(f"conf/server config has no model_config_file_path: {cfg.relative_to(ROOT)}")
            continue
        # model_config_file_path is relative to the server process working dir
        # (documented as _bin), e.g. ../conf/model/... actually points at the
        # repo root conf/model/...
        resolved = (ROOT / "_bin" / model_cfg_path).resolve()
        if not resolved.exists():
            errors.append(
                f"{cfg.relative_to(ROOT)} references missing model config: {model_cfg_path}"
            )
            continue
        if resolved.suffix != ".toml":
            errors.append(
                f"{cfg.relative_to(ROOT)} references non-toml model config: {model_cfg_path}"
            )
    return errors


def check_openapi_covers_server_uris() -> list[str]:
    """Every server_uri in conf/server must be declared in docs/openapi.json."""
    errors: list[str] = []
    openapi_path = ROOT / "docs" / "openapi.json"
    conf_server = ROOT / "conf" / "server"
    if not openapi_path.exists() or not conf_server.exists():
        return errors
    try:
        doc = json.loads(openapi_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(f"docs/openapi.json is not valid JSON: {exc}")
        return errors
    declared_paths = set(doc.get("paths", {}).keys())
    for cfg in sorted(conf_server.rglob("*.toml")):
        try:
            table = load_toml(cfg)
        except (ValueError, OSError):
            continue
        for section in table.values():
            if isinstance(section, dict) and "server_uri" in section:
                uri = section["server_uri"]
                if uri not in declared_paths:
                    errors.append(
                        f"server_uri {uri} (from {cfg.relative_to(ROOT)}) "
                        "is missing from docs/openapi.json paths"
                    )
    return errors


def check_openapi_doc_header_sync() -> list[str]:
    """src/server/openapi_doc.h must embed the exact docs/openapi.json content."""
    errors: list[str] = []
    json_path = ROOT / "docs" / "openapi.json"
    header_path = ROOT / "src" / "server" / "openapi_doc.h"
    if not json_path.exists() or not header_path.exists():
        errors.append("docs/openapi.json or src/server/openapi_doc.h is missing")
        return errors
    json_text = json_path.read_text(encoding="utf-8")
    header_text = header_path.read_text(encoding="utf-8")
    if json_text not in header_text:
        errors.append(
            "src/server/openapi_doc.h is out of sync with docs/openapi.json; "
            "run: python scripts/gen_openapi.py"
        )
    return errors


def check_demo_client_health() -> list[str]:
    """The demo client must be self-contained (config-driven, no orphan files).

    Guards against the historical breakage where test_server.py imported a
    `config_utils` module that did not exist in the repository, and against
    re-introducing the duplicated conf/py_demo config tree.
    """
    errors: list[str] = []

    # orphan guards: the demo client reads conf/server directly
    for orphan in [ROOT / "scripts" / "server" / "config_utils.py",
                   ROOT / "conf" / "py_demo"]:
        if orphan.exists():
            errors.append(
                f"orphan file must be removed: {orphan.relative_to(ROOT)} "
                "(demo client reads conf/server directly)"
            )

    locust = ROOT / "scripts" / "server" / "locust_performance.py"
    if locust.exists():
        errors.append(
            "orphan locust worker must be removed: scripts/server/locust_performance.py "
            "(use scripts/server/http_infer_rps.py)"
        )

    # syntax check for the catalog client and the HTTP serving-RPS client
    for script in [ROOT / "scripts" / "server" / "test_server.py",
                   ROOT / "scripts" / "server" / "http_infer_rps.py"]:
        if not script.exists():
            errors.append(f"missing demo client file: {script.relative_to(ROOT)}")
            continue
        try:
            py_compile.compile(str(script), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"demo client syntax error in {script.relative_to(ROOT)}: {exc}")

    # --list must work without network / requests / locust (CI runs this)
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "server" / "test_server.py"), "--list"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        errors.append(f"test_server.py --list could not run: {exc}")
        return errors
    if result.returncode != 0:
        errors.append(
            f"test_server.py --list failed (exit {result.returncode}): "
            f"{result.stdout.strip()} {result.stderr.strip()}"
        )

    try:
        selftest = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "server" / "http_infer_rps.py"), "--self-test"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        errors.append(f"http_infer_rps.py --self-test could not run: {exc}")
        return errors
    if selftest.returncode != 0:
        errors.append(
            f"http_infer_rps.py --self-test failed (exit {selftest.returncode}): "
            f"{selftest.stdout.strip()} {selftest.stderr.strip()}"
        )

    return errors


def check_server_config_structure() -> list[str]:
    """Every conf/server/*.toml must follow the canonical two-section layout:

    [XXX_SERVER]   -> server keys only (port/host/server_uri/worker_nums/...)
    [XXX]          -> model_config_file_path lives here, not in the server
                      section (the historical real_esrgan drift).
    """
    errors: list[str] = []
    conf_server = ROOT / "conf" / "server"
    if not conf_server.exists():
        return errors
    for cfg in sorted(conf_server.rglob("*.toml")):
        rel = cfg.relative_to(ROOT)
        try:
            table = load_toml(cfg)
        except (ValueError, OSError) as exc:
            errors.append(f"{rel} is not valid TOML: {exc}")
            continue
        server_sections = [sec for sec in table if sec.endswith("_SERVER")]
        model_sections = [sec for sec in table if not sec.endswith("_SERVER")]
        if len(server_sections) != 1:
            errors.append(
                f"{rel}: expected exactly one *_SERVER section, got {server_sections}"
            )
            continue
        server_kv = table[server_sections[0]]
        if isinstance(server_kv, dict) and "model_config_file_path" in server_kv:
            errors.append(
                f"{rel}: model_config_file_path must live in the [MODEL] subtable, "
                "not in the server section"
            )
        if not any(isinstance(table[ms], dict) and "model_config_file_path" in table[ms]
                   for ms in model_sections):
            errors.append(
                f"{rel}: missing model_config_file_path in a [MODEL] subtable"
            )
    return errors


_HTTP_ENTRY_RE = re.compile(
    r'\{\s*"([A-Z][A-Z0-9_]*)"\s*,\s*"[^"]*"\s*,\s*"[A-Z][A-Z0-9_]*_SERVER"'
)
_UNIFIED_SERVER_EXE = "mortred-model-server.out"


def parse_cpp_http_models() -> set[str]:
    """HTTP catalog ids: CvModelEntry rows that carry a *_SERVER section."""
    ids: set[str] = set()
    factory = ROOT / "src" / "factory"
    if not factory.exists():
        return ids
    for header in sorted(factory.glob("*_task.h")):
        text = header.read_text(encoding="utf-8")
        ids.update(_HTTP_ENTRY_RE.findall(text))
    return ids


def check_server_exe_mapping() -> list[str]:
    """conf/server `model=` <-> C++ HTTP catalog, unified server_exe only."""
    errors: list[str] = []
    conf_server = ROOT / "conf" / "server"
    if not conf_server.exists():
        return errors
    catalog_ids = parse_cpp_http_models()
    declared: set[str] = set()
    for cfg in sorted(conf_server.rglob("*.toml")):
        rel = cfg.relative_to(ROOT)
        try:
            table = load_toml(cfg)
        except (ValueError, OSError):
            continue
        server_sections = [sec for sec in table if sec.endswith("_SERVER")]
        model_sections = [sec for sec in table if not sec.endswith("_SERVER")]
        if len(server_sections) != 1:
            continue
        server_kv = table[server_sections[0]]
        if not isinstance(server_kv, dict):
            continue
        model = server_kv.get("model")
        if not model:
            errors.append(f"{rel}: missing model in [{server_sections[0]}]")
            continue
        if len(model_sections) == 1 and model != model_sections[0]:
            errors.append(
                f"{rel}: model={model!r} must equal non-_SERVER table [{model_sections[0]}]"
            )
        exe = server_kv.get("server_exe", _UNIFIED_SERVER_EXE)
        if exe != _UNIFIED_SERVER_EXE:
            errors.append(
                f"{rel}: server_exe must be {_UNIFIED_SERVER_EXE}, got {exe}"
            )
        declared.add(model)
        if catalog_ids and model not in catalog_ids:
            errors.append(f"{rel}: model={model} is not an HTTP catalog id")
    if catalog_ids:
        missing = sorted(catalog_ids - declared)
        if missing:
            errors.append(
                "HTTP catalog ids without a conf/server mapping: " + ", ".join(missing)
            )
    return errors


def check_trt_engine_manifest() -> list[str]:
    """Every engine referenced by conf/model [*_TRT] sections (any key ending
    with `model_file_path`) must be declared in conf/trt_engines.json, so the
    engine-regeneration script (scripts/convert_trt_engines.sh) can never miss
    a config-required engine."""
    errors: list[str] = []
    manifest_path = ROOT / "conf" / "trt_engines.json"
    if not manifest_path.exists():
        return errors
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(f"conf/trt_engines.json is not valid JSON: {exc}")
        return errors
    declared = {e.get("engine") for e in manifest.get("engines", [])}
    for cfg in sorted((ROOT / "conf" / "model").rglob("*.toml")):
        try:
            table = load_toml(cfg)
        except (ValueError, OSError):
            continue
        for sec, kv in table.items():
            if not sec.endswith("_TRT") or not isinstance(kv, dict):
                continue
            for key, value in kv.items():
                if not key.endswith("model_file_path") or not value:
                    continue
                resolved = (ROOT / "_bin" / value).resolve()
                try:
                    rel = resolved.relative_to(ROOT).as_posix()
                except ValueError:
                    continue
                if rel not in declared:
                    errors.append(
                        f"{cfg.relative_to(ROOT)} [{sec}] {key} engine {rel} "
                        "missing from conf/trt_engines.json"
                    )
    return errors


def check_factory_register_type_banned() -> list[str]:
    """src/factory/*_task.h must not call register_type: models construct
    directly; servers use the register_creator closure (avoids the
    "write the global registry on every create" anti-pattern)."""
    errors: list[str] = []
    for header in sorted((ROOT / "src" / "factory").glob("*_task.h")):
        for i, line in enumerate(header.read_text(encoding="utf-8").splitlines(), 1):
            if "register_type" in line:
                errors.append(
                    f"{header.relative_to(ROOT)}:{i}: register_type is banned in task "
                    f"headers (models construct directly; servers use register_creator)")
    return errors


def check_security_scan() -> list[str]:
    """Source-level security lint: bans dangerous calls
    (system/popen/strcpy/strcat/gets/scanf). Prevents historical vulnerability
    patterns from returning (review found zero hits repo-wide; enforced as a
    hard gate here)."""
    errors: list[str] = []
    banned = re.compile(r"\b(system|popen|strcpy|strcat|gets|scanf)\s*\(")
    src_root = ROOT / "src"
    if not src_root.exists():
        return errors
    for path in sorted(src_root.rglob("*")):
        if not path.is_file() or path.suffix not in (".h", ".hpp", ".inl", ".cpp", ".cc", ".c"):
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
                continue  # skip comment lines to reduce false positives
            if banned.search(line):
                errors.append(f"{path.relative_to(ROOT)}:{i}: banned call: {line.strip()}")
    return errors


def check_ci_no_python3_runs_sh() -> list[str]:
    """.github/workflows/*.yml must not run bash scripts via python3 (historical
    bug: ci.yml used `python3 scripts/convert_trt_engines.sh --list`, which made
    deploy-tools always fail)."""
    errors: list[str] = []
    wf_dir = ROOT / ".github" / "workflows"
    if not wf_dir.exists():
        return errors
    for wf in sorted(wf_dir.glob("*.yml")):
        for i, line in enumerate(wf.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"python3?\s+scripts/[^\s]+\.sh\b", line):
                errors.append(
                    f"{wf.relative_to(ROOT)}:{i}: running bash scripts via python3 is banned (use bash): {line.strip()}"
                )
    return errors


def check_scaffolder_task_metadata() -> list[str]:
    """Every field templates/model/tasks.json claims must exist in the real
    sources. This is the seam between the Python scaffolder and the C++
    catalogs: if either side is renamed, this check fails before a developer
    generates a model against stale metadata."""
    errors: list[str] = []
    manifest = ROOT / "templates" / "model" / "tasks.json"
    if not manifest.exists():
        return errors
    try:
        tasks = json.loads(manifest.read_text(encoding="utf-8")).get("tasks", {})
    except (json.JSONDecodeError, OSError) as exc:
        return [f"templates/model/tasks.json is not valid JSON: {exc}"]

    serializers = (ROOT / "src" / "server" / "response_serializers.h").read_text(encoding="utf-8")

    for task, spec in sorted(tasks.items()):
        where = f"tasks.json[{task}]"
        if not (ROOT / "src" / "models" / spec["model_dir"]).is_dir():
            errors.append(f"{where}: model_dir src/models/{spec['model_dir']} does not exist")
        io_header_path = spec.get("io_header")
        io_header = ROOT / "src" / io_header_path if io_header_path else None
        if io_header is None or not io_header.is_file():
            errors.append(f"{where}: io_header is missing or does not exist")
        else:
            io_text = io_header.read_text(encoding="utf-8")
            if not re.search(rf"namespace\s+{re.escape(spec['io_namespace'])}\s*\{{", io_text):
                errors.append(f"{where}: io_namespace {spec['io_namespace']} is not in {io_header_path}")
            # output types appear either as `struct clip_output {` or as a
            # `using std_*_output = ...` alias, so an identifier match is enough
            if not re.search(rf"\b{re.escape(spec['output_type'])}\b", io_text):
                errors.append(f"{where}: output_type {spec['output_type']} is not declared in {io_header_path}")

        catalog = ROOT / "src" / spec["catalog_header"]
        if not catalog.exists():
            errors.append(f"{where}: catalog_header {spec['catalog_header']} does not exist")
        else:
            catalog_text = catalog.read_text(encoding="utf-8")
            if f"{spec['catalog_function']}()" not in catalog_text:
                errors.append(
                    f"{where}: catalog_function {spec['catalog_function']}() "
                    f"is not defined in {spec['catalog_header']}"
                )

        filler = spec.get("response_filler")
        if filler and f"void {filler}(" not in serializers:
            errors.append(f"{where}: response_filler {filler} is not defined in response_serializers.h")
        if not filler and spec.get("server_section_suffix"):
            errors.append(f"{where}: model-only task must not declare server_section_suffix")

    return errors


def check_model_todo_markers() -> list[str]:
    """src/models/**.inl must use the single canonical scaffold marker. Other
    spellings (TODO:, FIXME) are banned so `grep -rn 'TODO(new_model)'` is a
    complete list of half-finished models."""
    errors: list[str] = []
    for path in sorted((ROOT / "src" / "models").rglob("*.inl")):
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if "TODO" not in line and "FIXME" not in line and "XXX" not in line:
                continue
            if "TODO(new_model)" in line:
                continue
            errors.append(f"{path.relative_to(ROOT)}:{i}: use TODO(new_model), not: {line.strip()}")
    return errors


def check_unique_catalog_listen() -> list[str]:
    """Ports / ids / URIs must be unique within one catalog profile, matching
    Catalog::init. CPU/GPU variants of the same model may share a port because
    only one profile is active per supervisor run."""
    errors: list[str] = []
    conf_server = ROOT / "conf" / "server"
    if not conf_server.is_dir():
        return errors
    rows: list[tuple[str, str, int, str, str]] = []
    for cfg in sorted(conf_server.rglob("*.toml")):
        rel = cfg.relative_to(ROOT).as_posix()
        try:
            table = load_toml(cfg)
        except (ValueError, OSError):
            continue
        server_sections = [sec for sec in table if sec.endswith("_SERVER")]
        if len(server_sections) != 1:
            continue
        kv = table[server_sections[0]]
        if not isinstance(kv, dict):
            continue
        profile = str(kv.get("profile") or "gpu")
        try:
            port = int(kv.get("port") or 0)
        except (TypeError, ValueError):
            continue
        uri = str(kv.get("server_uri") or "")
        mid = str(kv.get("model") or "")
        if port <= 0 or not uri.startswith("/") or not mid:
            continue
        rows.append((rel, profile, port, uri, mid))

    for runtime in ("gpu", "cpu"):
        seen_ports: dict[int, str] = {}
        seen_uris: dict[str, str] = {}
        seen_ids: dict[str, str] = {}
        for rel, profile, port, uri, mid in rows:
            if profile != "any" and profile != runtime:
                continue
            prev = seen_ports.get(port)
            if prev is not None:
                errors.append(
                    f"duplicate model server port {port} (profile={runtime}): "
                    f"{prev} and {rel}"
                )
            else:
                seen_ports[port] = rel
            prev = seen_uris.get(uri)
            if prev is not None:
                errors.append(
                    f"duplicate server_uri {uri} (profile={runtime}): "
                    f"{prev} and {rel}"
                )
            else:
                seen_uris[uri] = rel
            prev = seen_ids.get(mid)
            if prev is not None:
                errors.append(
                    f"duplicate server id {mid} (profile={runtime}): "
                    f"{prev} and {rel}"
                )
            else:
                seen_ids[mid] = rel
    return errors


def check_demo_pack() -> list[str]:
    """The shipped demo pack must only name HTTP catalog ids."""
    errors: list[str] = []
    pack = ROOT / "conf" / "packs" / "demo.toml"
    if not pack.is_file():
        return ["missing conf/packs/demo.toml"]
    catalog_ids: set[str] = set()
    conf_server = ROOT / "conf" / "server"
    if conf_server.is_dir():
        for cfg in conf_server.rglob("*.toml"):
            try:
                table = load_toml(cfg)
            except (ValueError, OSError):
                continue
            for kv in table.values():
                if isinstance(kv, dict):
                    model = kv.get("model")
                    if isinstance(model, str) and model:
                        catalog_ids.add(model)
    found = False
    for line in pack.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if t.startswith("[pack.") and t.endswith("]"):
            pid = t[6:-1]
            found = True
            if pid not in catalog_ids:
                errors.append(
                    f"conf/packs/demo.toml unknown catalog id [{pid}]"
                )
    if not found:
        errors.append("conf/packs/demo.toml has no [pack.<ID>] tables")
    return errors


def check_model_io_split() -> list[str]:
    """Guard the src/models/io/ split:
    - model_io_define.h stays a pure aggregate (includes only, no types), so
      nobody quietly adds code back to the compatibility header;
    - IO headers must not pull the whole OpenCV umbrella header: they only need
      Mat / Rect / Point / Size, all of which live in opencv2/core.hpp."""
    errors: list[str] = []
    aggregate = ROOT / "src" / "models" / "model_io_define.h"
    if aggregate.exists():
        for i, line in enumerate(aggregate.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith(("*", "/", "#")):
                continue
            errors.append(
                f"src/models/model_io_define.h:{i}: aggregate must only include models/io/*.h, got: {stripped}"
            )

    io_dir = ROOT / "src" / "models" / "io"
    if not io_dir.is_dir():
        return errors + ["src/models/io is missing"]
    for header in sorted(io_dir.glob("*.h")):
        for i, line in enumerate(header.read_text(encoding="utf-8").splitlines(), 1):
            if "opencv2/opencv.hpp" in line:
                errors.append(
                    f"{header.relative_to(ROOT)}:{i}: IO headers must include opencv2/core.hpp, not opencv.hpp"
                )
    return errors


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    errors.extend(check_layout_references())
    errors.extend(check_server_source_mapping())
    errors.extend(check_server_model_config_paths())
    errors.extend(check_openapi_covers_server_uris())
    errors.extend(check_openapi_doc_header_sync())
    errors.extend(check_server_config_structure())
    errors.extend(check_server_exe_mapping())
    errors.extend(check_trt_engine_manifest())
    errors.extend(check_factory_register_type_banned())
    errors.extend(check_security_scan())
    errors.extend(check_ci_no_python3_runs_sh())
    errors.extend(check_scaffolder_task_metadata())
    errors.extend(check_model_todo_markers())
    errors.extend(check_model_io_split())
    errors.extend(check_demo_client_health())
    errors.extend(check_unique_catalog_listen())
    errors.extend(check_demo_pack())
    from check_hosted_golden import check_ci_inference_contract

    errors.extend(check_ci_inference_contract())
    if args.check_stale_binaries:
        errors.extend(check_stale_binaries())

    if errors:
        print("Repository consistency check failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("Repository consistency check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
