#!/usr/bin/env python3
"""Black-box acceptance for the classification scaffolder (P1-1).

Generates the fixed probe identity, temporarily registers its output-contract
unittest, builds and runs it, asserts unfilled hooks return
MODEL_NOT_IMPLEMENTED, asserts --list omits the probe, then deletes every
generated artifact and restores test/CMakeLists.txt — including on failure.

    python3 scripts/verify_cls_scaffold_probe.py

Exit 0 if and only if acceptance criteria 2–6 hold. Never passes --apply-catalog.
"""

from __future__ import annotations

import atexit
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_ROOT / "build" / "full"
CMAKE_LISTS = REPO_ROOT / "test" / "CMakeLists.txt"

TASK = "classification"
NAME = "p1_1_cls_probe"
CLASS_NAME = "P11ClsProbe"
BACKEND = "mnn"
FILE_BASE = "p11_cls_probe"
SECTION = "P1_1_CLS_PROBE"
TEST_TARGET = f"{FILE_BASE}_output_contract_unittest"

MARKER_BEGIN = f"# >>> {NAME} temporary (do not commit)"
MARKER_END = f"# <<< {NAME} temporary"

# Paths new_model.py writes for this probe (must match scripts/new_model.py plan).
GENERATED = [
    REPO_ROOT / "src" / "models" / "classification" / f"{FILE_BASE}.h",
    REPO_ROOT / "src" / "models" / "classification" / f"{FILE_BASE}.inl",
    REPO_ROOT / "conf" / "model" / "classification" / NAME / f"{NAME}_config.toml",
    REPO_ROOT / "conf" / "server" / "classification" / NAME / f"{NAME}_server_config.toml",
    REPO_ROOT / "test" / f"{TEST_TARGET}.cc",
    REPO_ROOT / "docs" / "models" / "classification" / f"{NAME}.md",
]

_cmake_backup: str | None = None
_cleanup_done = False


def log(msg: str) -> None:
    print(f"[cls-scaffold-probe] {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"[cls-scaffold-probe] FAIL: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    lib_dirs = [str(REPO_ROOT / "_lib"), str(REPO_ROOT / "3rd_party" / "libs")]
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
    return env


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    log("$ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd or REPO_ROOT, text=True, capture_output=True, env=runtime_env())
    if result.stdout:
        sys.stdout.write(result.stdout)
        if not result.stdout.endswith("\n"):
            sys.stdout.write("\n")
    if result.stderr:
        sys.stderr.write(result.stderr)
        if not result.stderr.endswith("\n"):
            sys.stderr.write("\n")
    if check and result.returncode != 0:
        fail(f"command failed (exit {result.returncode}): {' '.join(cmd)}")
    return result


def restore_cmake() -> None:
    global _cmake_backup
    if _cmake_backup is None:
        return
    CMAKE_LISTS.write_text(_cmake_backup, encoding="utf-8", newline="\n")
    _cmake_backup = None
    log(f"restored {CMAKE_LISTS.relative_to(REPO_ROOT)}")


def remove_generated() -> None:
    for path in GENERATED:
        if path.is_file():
            path.unlink()
            log(f"removed {path.relative_to(REPO_ROOT)}")
    for directory in (
        REPO_ROOT / "conf" / "model" / "classification" / NAME,
        REPO_ROOT / "conf" / "server" / "classification" / NAME,
    ):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
            log(f"removed empty {directory.relative_to(REPO_ROOT)}")


def cleanup() -> None:
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    restore_cmake()
    remove_generated()


def patch_cmake() -> None:
    global _cmake_backup
    text = CMAKE_LISTS.read_text(encoding="utf-8")
    if MARKER_BEGIN in text:
        fail(f"{CMAKE_LISTS} already contains a temporary {NAME} hook; aborting")
    _cmake_backup = text

    list_anchor = "    list(APPEND TEST_LIST rtdetr_detector_output_contract_unittest)\n"
    if list_anchor not in text:
        fail("could not find rtdetr TEST_LIST anchor in test/CMakeLists.txt")
    list_block = (
        f"{list_anchor}"
        f"    {MARKER_BEGIN}\n"
        f"    list(APPEND TEST_LIST {TEST_TARGET})\n"
        f"    {MARKER_END}\n"
    )
    text = text.replace(list_anchor, list_block, 1)

    elseif_anchor = (
        '    elseif(${src} STREQUAL "rtdetr_detector_output_contract_unittest")\n'
        "        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn)\n"
    )
    if elseif_anchor not in text:
        fail("could not find rtdetr EXTRA_LIBS anchor in test/CMakeLists.txt")
    elseif_block = (
        f"{elseif_anchor}"
        f"    {MARKER_BEGIN}\n"
        f'    elseif(${{src}} STREQUAL "{TEST_TARGET}")\n'
        f"        set(EXTRA_LIBS models glog::glog ${{OpenCV_LIBS}} vendored::mnn)\n"
        f"    {MARKER_END}\n"
    )
    text = text.replace(elseif_anchor, elseif_block, 1)
    CMAKE_LISTS.write_text(text, encoding="utf-8", newline="\n")
    log(f"temporarily registered {TEST_TARGET} in test/CMakeLists.txt")


def assert_not_implemented_hooks(inl: Path) -> None:
    body = inl.read_text(encoding="utf-8")
    if body.count("return StatusCode::MODEL_NOT_IMPLEMENTED;") < 2:
        fail(f"{inl.name}: expected MODEL_NOT_IMPLEMENTED in on_init and postprocess")
    if re.search(r"return\s+StatusCode::OK\b", body):
        fail(f"{inl.name}: scaffold must not return OK from hooks")


def assert_list_omits_probe(list_out: str) -> None:
    lowered = list_out.lower()
    for needle in (NAME, SECTION, CLASS_NAME, FILE_BASE):
        if needle.lower() in lowered:
            fail(f"--list output unexpectedly contains {needle!r}")


def assert_catalog_clean() -> None:
    catalog = REPO_ROOT / "src" / "factory" / "classification_task.h"
    text = catalog.read_text(encoding="utf-8")
    for needle in (NAME, SECTION, CLASS_NAME, FILE_BASE, "create_p1_1_cls_probe"):
        if needle in text:
            fail(f"{catalog.relative_to(REPO_ROOT)} still mentions probe identity {needle!r}")


def find_binary(name: str) -> Path:
    candidates = [
        REPO_ROOT / "_bin" / name,
        BUILD_DIR / "bin" / name,
        BUILD_DIR / "test" / name,
        BUILD_DIR / name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    fail(f"built binary not found for {name}; tried: " + ", ".join(str(p) for p in candidates))
    raise AssertionError("unreachable")


def main() -> int:
    atexit.register(cleanup)
    if not (REPO_ROOT / "scripts" / "new_model.py").is_file():
        fail("run from repository root (scripts/new_model.py missing)")

    leftovers = [p for p in GENERATED if p.exists()]
    if leftovers:
        fail(
            "probe artifacts already exist; remove them first:\n  "
            + "\n  ".join(str(p.relative_to(REPO_ROOT)) for p in leftovers)
        )
    if MARKER_BEGIN in CMAKE_LISTS.read_text(encoding="utf-8"):
        fail("test/CMakeLists.txt still has a temporary probe hook; restore it first")

    # 1) Generate scaffold (no --dry-run, no --apply-catalog).
    run(
        [
            sys.executable,
            "scripts/new_model.py",
            "--task",
            TASK,
            "--name",
            NAME,
            "--class",
            CLASS_NAME,
            "--backend",
            BACKEND,
        ]
    )
    missing = [p for p in GENERATED if not p.is_file()]
    if missing:
        fail(
            "scaffolder did not create expected files:\n  "
            + "\n  ".join(str(p.relative_to(REPO_ROOT)) for p in missing)
        )
    assert_not_implemented_hooks(REPO_ROOT / "src" / "models" / "classification" / f"{FILE_BASE}.inl")
    assert_catalog_clean()

    # 2) Temporary CMake hook + reconfigure + build.
    patch_cmake()
    run(["cmake", "--preset", "full"])
    jobs = str(os.cpu_count() or 4)
    run(
        [
            "cmake",
            "--build",
            str(BUILD_DIR),
            "--target",
            TEST_TARGET,
            "mortred-model-server.out",
            f"-j{jobs}",
        ]
    )

    test_bin = find_binary(TEST_TARGET)
    server_bin = find_binary("mortred-model-server.out")

    # 3) Contract test must pass; glog should show the unimplemented postprocess path.
    test_result = run([str(test_bin)], cwd=REPO_ROOT)
    combined = (test_result.stdout or "") + (test_result.stderr or "")
    if f"{SECTION}::postprocess is not implemented yet" not in combined:
        fail("contract test did not exercise MODEL_NOT_IMPLEMENTED postprocess path")

    # 4) --list must omit the probe while generated sources still exist.
    listed = run([str(server_bin), "--list"], cwd=REPO_ROOT)
    assert_list_omits_probe(listed.stdout)
    log("--list omits probe while scaffold files exist")

    # 5) Cleanup, then consistency + catalog still clean.
    cleanup()
    atexit.unregister(cleanup)

    still_there = [p for p in GENERATED if p.exists()]
    if still_there:
        fail(
            "cleanup left probe artifacts:\n  "
            + "\n  ".join(str(p.relative_to(REPO_ROOT)) for p in still_there)
        )
    if MARKER_BEGIN in CMAKE_LISTS.read_text(encoding="utf-8"):
        fail("cleanup left temporary CMake markers")
    assert_catalog_clean()

    listed_after = run([str(server_bin), "--list"], cwd=REPO_ROOT)
    assert_list_omits_probe(listed_after.stdout)

    run([sys.executable, "scripts/check_consistency.py"])
    log("PASS: classification scaffold probe acceptance 2–6")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except SystemExit as exc:
        cleanup()
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[cls-scaffold-probe] ERROR: {exc}", file=sys.stderr)
        cleanup()
        sys.exit(1)
    sys.exit(code)
