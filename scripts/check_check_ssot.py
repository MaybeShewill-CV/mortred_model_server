#!/usr/bin/env python3
"""Fail-closed SSOT for what default `check` does not run.

test/CHECK_SSOT.md lists every ctest name excluded from
`cmake --build <dir> --target check`, why, and the alternate gate.
test/CMakeLists.txt labels and test/run_default_check.cmake must match.
README and docs/ci-golden-regression.md must not advertise those names as
covered by default check. Hosted golden stays on MORTRED_CI_REQUIRE_WEIGHTS
plus scripts/ci_assert_gtest_xml.py.

Usage:
  python3 scripts/check_check_ssot.py
  python3 scripts/check_check_ssot.py --self-test
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SSOT = ROOT / "test" / "CHECK_SSOT.md"
CMAKE = ROOT / "test" / "CMakeLists.txt"
DRIVER = ROOT / "test" / "run_default_check.cmake"
README = ROOT / "README.md"
README_ZH = ROOT / "README.zh-cn.md"
GOLDEN_DOC = ROOT / "docs" / "ci-golden-regression.md"
GOLDEN_SKIP = ROOT / "test" / "GOLDEN_SKIP.md"
CI = ROOT / ".github" / "workflows" / "ci.yml"
LEDGER = ROOT / "cowork" / "ledger" / "index.md"

CLAIM_RE = re.compile(
    r"`check`|--target check|target check|default check|默认 check|默认`check`"
)
EXCLUSION_MARKERS = (
    "heavy-golden",
    "excluded",
    "excludes",
    "does not run",
    "not part of",
    "排除",
    "不包含",
    "不在默认",
    "CHECK_SSOT",
)
HEAVY_GATE_TOKENS = (
    "MORTRED_CI_REQUIRE_WEIGHTS",
    "ci_assert_gtest_xml.py",
    "MORTRED_CHECK_HEAVY",
)
CI_TOKENS = (
    "check_check_ssot.py",
    "MORTRED_CI_REQUIRE_WEIGHTS=1",
    "ci_assert_gtest_xml.py",
    "./build-cpu/bin/model_golden_test",
    "./build-gpu/bin/model_golden_test",
)
GOLDEN_DOC_TOKENS = (
    "test/CHECK_SSOT.md",
    "heavy-golden",
    "does not run",
    "model_golden_test",
    "MORTRED_CI_REQUIRE_WEIGHTS",
    "ci_assert_gtest_xml.py",
    "MORTRED_CHECK_HEAVY",
)


@dataclass
class Bundle:
    ssot: str
    cmake: str
    driver: str
    readme: str
    readme_zh: str
    golden_doc: str
    golden_skip: str
    ci: str
    ledger: str


def parse_ssot(text: str) -> tuple[list[dict[str, str]], list[str]]:
    errors: list[str] = []
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not cells:
            continue
        head = cells[0].replace("`", "").strip()
        if head in {"ctest name", ""} or set(head) <= set("-: "):
            continue
        if len(cells) != 4:
            errors.append(
                f"test/CHECK_SSOT.md:{lineno}: expected 4 columns, got {len(cells)}"
            )
            continue
        name, label, why, gate = (c.replace("`", "").strip() for c in cells)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", name):
            errors.append(f"test/CHECK_SSOT.md:{lineno}: bad ctest name {name!r}")
            continue
        if label not in {"heavy-golden", "valgrind"}:
            errors.append(
                f"test/CHECK_SSOT.md:{lineno}: label {label!r} is not heavy-golden or valgrind"
            )
        if not why:
            errors.append(f"test/CHECK_SSOT.md:{lineno}: {name} is missing why")
        if not gate:
            errors.append(f"test/CHECK_SSOT.md:{lineno}: {name} is missing alternate gate")
        if label == "heavy-golden":
            for token in HEAVY_GATE_TOKENS:
                if token not in gate:
                    errors.append(
                        f"test/CHECK_SSOT.md:{lineno}: {name} alternate gate missing {token}"
                    )
        elif label == "valgrind" and "ctest -L valgrind" not in gate:
            errors.append(
                f"test/CHECK_SSOT.md:{lineno}: {name} alternate gate must name "
                "ctest -L valgrind"
            )
        if name in seen:
            errors.append(f"test/CHECK_SSOT.md:{lineno}: duplicate ctest name {name}")
        seen.add(name)
        rows.append({"name": name, "label": label, "why": why, "gate": gate})
    if not any(row["label"] == "heavy-golden" for row in rows):
        errors.append("test/CHECK_SSOT.md: no heavy-golden rows")
    if not any(row["label"] == "valgrind" for row in rows):
        errors.append("test/CHECK_SSOT.md: no valgrind rows")
    return rows, errors


def cmake_string_list(text: str, var: str) -> tuple[list[str], list[str]]:
    matches = re.findall(rf"set\(\s*{var}\b(.*?)\)", text, flags=re.S)
    if len(matches) != 1:
        return [], [f"test/CMakeLists.txt: expected one set({var}), found {len(matches)}"]
    names: list[str] = []
    for line in matches[0].splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.extend(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", line))
    if not names:
        return [], [f"test/CMakeLists.txt: set({var}) is empty"]
    return names, []


def check_labels(cmake: str, rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    heavy, heavy_err = cmake_string_list(cmake, "MORTRED_HEAVY_GOLDEN_TESTS")
    valgrind, valgrind_err = cmake_string_list(cmake, "MORTRED_VALGRIND_TEST_TARGETS")
    errors.extend(heavy_err)
    errors.extend(valgrind_err)
    ssot_heavy = [row["name"] for row in rows if row["label"] == "heavy-golden"]
    ssot_valgrind = [row["name"] for row in rows if row["label"] == "valgrind"]
    cmake_valgrind = [f"{name}-memory-check" for name in valgrind]
    if heavy and set(heavy) != set(ssot_heavy):
        errors.append(
            "heavy-golden mismatch: "
            f"CMake {sorted(heavy)} vs CHECK_SSOT {sorted(ssot_heavy)}"
        )
    if valgrind and set(cmake_valgrind) != set(ssot_valgrind):
        errors.append(
            "valgrind mismatch: "
            f"CMake {sorted(cmake_valgrind)} vs CHECK_SSOT {sorted(ssot_valgrind)}"
        )
    label_hits = re.findall(r'LABELS\s+"heavy-golden"', cmake)
    if len(label_hits) != 1:
        errors.append(
            f'test/CMakeLists.txt: expected exactly one LABELS "heavy-golden", '
            f"found {len(label_hits)}"
        )
    block = re.search(
        r"foreach\(heavy \$\{MORTRED_HEAVY_GOLDEN_TESTS\}\)(.*?)endforeach\(\)",
        cmake,
        flags=re.S,
    )
    if block is None:
        errors.append(
            "test/CMakeLists.txt: missing foreach(heavy ${MORTRED_HEAVY_GOLDEN_TESTS})"
        )
    else:
        body = block.group(1)
        for token in ("RUN_SERIAL TRUE", 'LABELS "heavy-golden"', "SKIP_RETURN_CODE 77"):
            if token not in body:
                errors.append(f"test/CMakeLists.txt: heavy-golden loop missing {token}")
        if "if(NOT TEST ${heavy})" not in body:
            errors.append(
                "test/CMakeLists.txt: heavy-golden loop must fail if a listed test "
                "was not registered"
            )
    if 'LABELS "valgrind"' not in cmake:
        errors.append('test/CMakeLists.txt: missing LABELS "valgrind"')
    if "run_default_check.cmake" not in cmake:
        errors.append("test/CMakeLists.txt: check target must call run_default_check.cmake")
    if re.search(r"add_custom_target\(check COMMAND \$\{CMAKE_CTEST_COMMAND\}", cmake):
        errors.append(
            "test/CMakeLists.txt: check must not invoke ctest directly; "
            "use run_default_check.cmake so MORTRED_CHECK_HEAVY is honored"
        )
    return errors


def check_driver(text: str) -> list[str]:
    errors: list[str] = []
    if 'set(_exclude "valgrind|heavy-golden")' not in text:
        errors.append(
            'run_default_check.cmake: default exclude must be set(_exclude "valgrind|heavy-golden")'
        )
    if 'set(_exclude "valgrind")' not in text:
        errors.append(
            'run_default_check.cmake: MORTRED_CHECK_HEAVY=1 branch must set(_exclude "valgrind")'
        )
    if 'if(_heavy STREQUAL "1")' not in text or "MORTRED_CHECK_HEAVY" not in text:
        errors.append("run_default_check.cmake: must treat only MORTRED_CHECK_HEAVY=1 as opt-in")
    if "-LE" not in text:
        errors.append("run_default_check.cmake: must pass -LE to ctest")
    return errors


def check_claims(path: str, text: str, names: list[str]) -> list[str]:
    errors: list[str] = []
    if "test/CHECK_SSOT.md" not in text:
        errors.append(f"{path}: must point at test/CHECK_SSOT.md")
    paragraphs = re.split(r"\n\s*\n", text)
    for para in paragraphs:
        if not CLAIM_RE.search(para):
            continue
        if any(marker in para for marker in EXCLUSION_MARKERS):
            continue
        for name in names:
            if re.search(rf"(?<![A-Za-z0-9_.]){re.escape(name)}(?![A-Za-z0-9_])", para):
                errors.append(
                    f"{path}: {name} is advertised next to default check without "
                    "an exclusion marker"
                )
    return errors


def check_golden_doc(text: str) -> list[str]:
    errors: list[str] = []
    for token in GOLDEN_DOC_TOKENS:
        if token not in text:
            errors.append(f"docs/ci-golden-regression.md: missing {token}")
    if "skip-as-pass for the rest of the zoo" in text:
        errors.append(
            "docs/ci-golden-regression.md: still says default check skip-as-pass "
            "covers the rest of the golden zoo"
        )
    return errors


def check_golden_skip(text: str) -> list[str]:
    errors: list[str] = []
    for token in ("CHECK_SSOT.md", "heavy-golden", "MORTRED_CI_REQUIRE_WEIGHTS"):
        if token not in text:
            errors.append(f"test/GOLDEN_SKIP.md: missing {token}")
    return errors


def check_ci(text: str) -> list[str]:
    errors: list[str] = []
    for token in CI_TOKENS:
        if token not in text:
            errors.append(f".github/workflows/ci.yml: missing {token}")
    return errors


def check_ledger(text: str) -> list[str]:
    errors: list[str] = []
    if "A4 未做" in text:
        errors.append(
            'cowork/ledger/index.md: still says "A4 未做"; point at test/CHECK_SSOT.md'
        )
    for token in ("test/CHECK_SSOT.md", "heavy-golden"):
        if token not in text:
            errors.append(f"cowork/ledger/index.md: missing {token}")
    return errors


def evaluate(bundle: Bundle) -> list[str]:
    rows, errors = parse_ssot(bundle.ssot)
    names = [row["name"] for row in rows]
    errors.extend(check_labels(bundle.cmake, rows))
    errors.extend(check_driver(bundle.driver))
    errors.extend(check_claims("README.md", bundle.readme, names))
    errors.extend(check_claims("README.zh-cn.md", bundle.readme_zh, names))
    errors.extend(check_claims("docs/ci-golden-regression.md", bundle.golden_doc, names))
    errors.extend(check_golden_doc(bundle.golden_doc))
    errors.extend(check_golden_skip(bundle.golden_skip))
    errors.extend(check_ci(bundle.ci))
    errors.extend(check_ledger(bundle.ledger))
    return errors


def load_bundle() -> Bundle:
    return Bundle(
        ssot=SSOT.read_text(encoding="utf-8"),
        cmake=CMAKE.read_text(encoding="utf-8"),
        driver=DRIVER.read_text(encoding="utf-8"),
        readme=README.read_text(encoding="utf-8"),
        readme_zh=README_ZH.read_text(encoding="utf-8"),
        golden_doc=GOLDEN_DOC.read_text(encoding="utf-8"),
        golden_skip=GOLDEN_SKIP.read_text(encoding="utf-8"),
        ci=CI.read_text(encoding="utf-8"),
        ledger=LEDGER.read_text(encoding="utf-8"),
    )


def _synthetic_bundle() -> Bundle:
    ssot = """# synthetic

Default `check` excludes heavy-golden. See this file.

| ctest name | label | why | alternate gate |
| --- | --- | --- | --- |
| model_golden_test | heavy-golden | oom | MORTRED_CI_REQUIRE_WEIGHTS and ci_assert_gtest_xml.py and MORTRED_CHECK_HEAVY |
| status_code_unittest-memory-check | valgrind | curated | ctest -L valgrind |
"""
    cmake = """
set(MORTRED_HEAVY_GOLDEN_TESTS
    model_golden_test
)
set(MORTRED_VALGRIND_TEST_TARGETS
    status_code_unittest
)
add_custom_target(check
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/run_default_check.cmake)
if(MORTRED_BUILD_FULL)
    foreach(heavy ${MORTRED_HEAVY_GOLDEN_TESTS})
        if(NOT TEST ${heavy})
            message(FATAL_ERROR "missing ${heavy}")
        endif()
        set_tests_properties(${heavy} PROPERTIES
            SKIP_RETURN_CODE 77
            RUN_SERIAL TRUE
            LABELS "heavy-golden")
    endforeach()
endif()
set_tests_properties(${src}-memory-check PROPERTIES LABELS "valgrind")
"""
    driver = """
set(_heavy "$ENV{MORTRED_CHECK_HEAVY}")
if(_heavy STREQUAL "1")
    set(_exclude "valgrind")
else()
    set(_exclude "valgrind|heavy-golden")
endif()
set(_cmd "${CTEST}" --output-on-failure -LE "${_exclude}")
"""
    readme = (
        "Default `check` excludes heavy-golden and does not run `model_golden_test`.\n"
        "See test/CHECK_SSOT.md.\n"
    )
    golden_doc = (
        "Default `check` excludes heavy-golden and does not run model_golden_test. "
        "test/CHECK_SSOT.md. MORTRED_CI_REQUIRE_WEIGHTS. ci_assert_gtest_xml.py. "
        "MORTRED_CHECK_HEAVY.\n"
    )
    golden_skip = "CHECK_SSOT.md heavy-golden MORTRED_CI_REQUIRE_WEIGHTS\n"
    ci = " ".join(CI_TOKENS) + "\n"
    ledger = "test/CHECK_SSOT.md heavy-golden\n"
    return Bundle(
        ssot=ssot,
        cmake=cmake,
        driver=driver,
        readme=readme,
        readme_zh=readme,
        golden_doc=golden_doc,
        golden_skip=golden_skip,
        ci=ci,
        ledger=ledger,
    )


def self_test_logic() -> list[str]:
    errors: list[str] = []
    good = _synthetic_bundle()
    if evaluate(good):
        errors.append(f"self-test: synthetic bundle should pass: {evaluate(good)}")
    drifted = _synthetic_bundle()
    drifted.cmake = drifted.cmake.replace("model_golden_test", "not_the_golden", 1)
    drift_errors = evaluate(drifted)
    if not any("heavy-golden mismatch" in item for item in drift_errors):
        errors.append(f"self-test: cmake list drift must fail: {drift_errors}")
    claimed = _synthetic_bundle()
    claimed.readme += "\nDefault `check` runs model_golden_test and that is the suite.\n"
    claim_errors = evaluate(claimed)
    if not any("advertised next to default check" in item for item in claim_errors):
        errors.append(f"self-test: README claim must fail: {claim_errors}")
    stale = _synthetic_bundle()
    stale.ledger += "A4 未做\n"
    if not any("A4 未做" in item for item in evaluate(stale)):
        errors.append("self-test: ledger A4 未做 must fail")
    return errors


def self_test_driver() -> list[str]:
    errors: list[str] = []
    cmake = shutil.which("cmake")
    ctest = shutil.which("ctest")
    if not cmake or not ctest:
        return ["self-test: cmake and ctest must be on PATH to prove the check driver"]
    if not DRIVER.is_file():
        return ["self-test: test/run_default_check.cmake is missing"]
    with tempfile.TemporaryDirectory(prefix="mortred-check-ssot-") as td:
        tmp = Path(td)
        ran = tmp / "ran.txt"
        writer = tmp / "mark.py"
        writer.write_text(
            "import sys\n"
            "from pathlib import Path\n"
            "Path(sys.argv[2]).open('a', encoding='utf-8').write(sys.argv[1] + '\\n')\n",
            encoding="utf-8",
        )
        py = sys.executable
        (tmp / "CTestTestfile.cmake").write_text(
            "\n".join(
                [
                    f'add_test(normal "{py}" "{writer}" normal "{ran}")',
                    f'add_test(heavy "{py}" "{writer}" heavy "{ran}")',
                    'set_tests_properties(heavy PROPERTIES LABELS "heavy-golden")',
                    f'add_test(mem "{py}" "{writer}" mem "{ran}")',
                    'set_tests_properties(mem PROPERTIES LABELS "valgrind")',
                    "",
                ]
            ),
            encoding="utf-8",
        )

        def invoke(heavy: str | None) -> tuple[int, list[str], str]:
            if ran.exists():
                ran.unlink()
            env = os.environ.copy()
            env.pop("MORTRED_CHECK_HEAVY", None)
            if heavy is not None:
                env["MORTRED_CHECK_HEAVY"] = heavy
            proc = subprocess.run(
                [
                    cmake,
                    f"-DCTEST={ctest}",
                    f"-DTEST_DIR={tmp}",
                    "-DJUNIT=",
                    "-P",
                    str(DRIVER),
                ],
                cwd=td,
                env=env,
                capture_output=True,
                text=True,
            )
            lines = ran.read_text(encoding="utf-8").splitlines() if ran.exists() else []
            return proc.returncode, lines, proc.stderr

        code, lines, err = invoke(None)
        if code != 0 or lines != ["normal"]:
            errors.append(
                f"self-test: default check should run only normal, got code={code} lines={lines} stderr={err}"
            )
        code, lines, err = invoke("0")
        if code != 0 or lines != ["normal"]:
            errors.append(
                f"self-test: MORTRED_CHECK_HEAVY=0 should exclude heavy-golden, got code={code} lines={lines}"
            )
        code, lines, err = invoke("1")
        if code != 0 or lines != ["normal", "heavy"]:
            errors.append(
                f"self-test: MORTRED_CHECK_HEAVY=1 should run normal and heavy, not mem, "
                f"got code={code} lines={lines} stderr={err}"
            )
    return errors


def self_test() -> int:
    errors = self_test_logic()
    errors.extend(self_test_driver())
    if errors:
        for item in errors:
            print(f"FAIL: {item}", file=sys.stderr)
        print("SELF-TEST FAIL")
        return 1
    print("SELF-TEST PASS")
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()
    missing = [
        path
        for path in (SSOT, CMAKE, DRIVER, README, README_ZH, GOLDEN_DOC, GOLDEN_SKIP, CI, LEDGER)
        if not path.is_file()
    ]
    if missing:
        for path in missing:
            print(f"ERROR: missing {path.relative_to(ROOT)}", file=sys.stderr)
        return 1
    errors = evaluate(load_bundle())
    if errors:
        for item in errors:
            print(f"ERROR: {item}", file=sys.stderr)
        print("check SSOT drifted", file=sys.stderr)
        return 1
    print("check SSOT agrees: default check excludes heavy-golden and valgrind")
    return 0


if __name__ == "__main__":
    sys.exit(main())
