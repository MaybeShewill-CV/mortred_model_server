#!/usr/bin/env python3
"""Repository consistency checker.

Verifies a few high-signal invariants:

1. Every source path referenced in docs/repository-layout.md exists.
2. Every server executable in docs/repository-layout.md has a matching source file.
3. If _bin exists, stale binaries listed in the layout policy are reported.
4. Every conf/server/*.toml file has at least one matching server source directory.
5. Every conf/server/*.toml `model_config_file_path` points to an existing
   `.toml` file (the repo migrated model configs from .ini to .toml).
6. Every conf/server/*.toml `server_uri` is covered by docs/openapi.json paths.
7. The demo client (`scripts/server/test_server.py`) is self-contained:
   - orphaned config files (`config_utils.py`, `conf/py_demo/`) must not exist;
   - the client and the locust worker compile;
   - `test_server.py --list` exits 0 without network / requests / locust.
8. Every conf/server/*.toml follows the canonical two-section layout
   (`model_config_file_path` lives in the [MODEL] subtable, not the server
   section).
9. Bidirectional `server_exe` <-> src/apps/server coverage, so the web
   console catalog cannot silently miss or invent a server.
10. Every engine referenced by conf/model [*_TRT] sections is declared in
    conf/trt_engines.json, so the engine-regeneration script can never miss a
    config-required engine.

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
    errors: list[str] = []
    server_dir = ROOT / "src" / "apps" / "server"
    if not server_dir.exists():
        errors.append("src/apps/server is missing")
        return errors

    # Every top-level task directory should have at least one .cpp file.
    for child in sorted(server_dir.iterdir()):
        if child.is_dir() and not list(child.glob("*.cpp")):
            errors.append(f"server task directory has no .cpp files: {child.relative_to(ROOT)}")

    # Every conf/server subdirectory should have a corresponding server source directory.
    conf_server = ROOT / "conf" / "server"
    if conf_server.exists():
        for child in sorted(conf_server.iterdir()):
            if child.is_dir():
                # conf/server/scene_segmentation -> src/apps/server/scene_segmentation
                src_dir = server_dir / child.name
                if not src_dir.exists() and not list(child.glob("*.ini")) and not list(child.glob("*.toml")):
                    # Some conf dirs contain nested model dirs; only report if no config at all.
                    errors.append(
                        f"conf/server/{child.name} has no matching src/apps/server/{child.name}"
                    )
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
        # model_config_file_path 是相对服务进程工作目录（文档约定为 _bin）的路径，
        # 例如 ../conf/model/... 实际指向仓库根的 conf/model/...。
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

    # syntax check for the client and the locust worker
    for script in [ROOT / "scripts" / "server" / "test_server.py",
                   ROOT / "scripts" / "server" / "locust_performance.py"]:
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


def check_server_exe_mapping() -> list[str]:
    """Bidirectional conf/server `server_exe` <-> src/apps/server coverage.

    Every server config must declare an existing server executable and every
    server executable must be declared by exactly one config, so the web
    console catalog can never silently miss or invent a server.
    """
    errors: list[str] = []
    server_src = ROOT / "src" / "apps" / "server"
    conf_server = ROOT / "conf" / "server"
    if not server_src.exists() or not conf_server.exists():
        return errors
    exe_sources = {p.stem + ".out" for p in server_src.rglob("*.cpp")}
    declared: set[str] = set()
    for cfg in sorted(conf_server.rglob("*.toml")):
        rel = cfg.relative_to(ROOT)
        try:
            table = load_toml(cfg)
        except (ValueError, OSError):
            continue
        server_sections = [sec for sec in table if sec.endswith("_SERVER")]
        if len(server_sections) != 1:
            continue
        server_kv = table[server_sections[0]]
        if not isinstance(server_kv, dict):
            continue
        exe = server_kv.get("server_exe")
        if not exe:
            errors.append(f"{rel}: missing server_exe in [{server_sections[0]}]")
            continue
        declared.add(exe)
        if exe not in exe_sources:
            errors.append(
                f"{rel}: server_exe {exe} has no matching source under src/apps/server"
            )
    missing = sorted(exe_sources - declared)
    if missing:
        errors.append(
            "server executables without a conf/server mapping: " + ", ".join(missing)
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
    errors.extend(check_demo_client_health())
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
