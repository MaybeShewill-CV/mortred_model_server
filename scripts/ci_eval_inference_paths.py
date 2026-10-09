#!/usr/bin/env python3
"""Decide the inference-paths check from cpu-profile and gpu-pr-gate results.

Exit 2 means cpu-profile succeeded and gpu-pr-gate was skipped: there is no
TensorRT or CUDA proof. Exit 0 means both succeeded. Exit 1 means cpu-profile
did not succeed, or the GPU smoke actually failed or was cancelled, or the
GPU result is anything else. cpu-profile is judged before the skipped case.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import ci_print_tier_summary  # noqa: E402

ROOT = SCRIPT_DIR.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

CPU_MUST_SUCCEED = (
    "::error::cpu-profile must succeed (fork and maintainer hosted CPU goldens)"
)
NOT_A_TRT_PROOF = "NOT_A_TRT_PROOF"
TRT_PROOF_YES = "TRT_PROOF=yes"
NOT_A_PROOF_SENTENCE = "This result is not a TensorRT or CUDA proof."
INLINE_SKIP_AS_SUCCESS = re.compile(
    r"success[\s\S]*skipped[\s\S]*exit 0|skipped[\s\S]*success[\s\S]*exit 0"
)


def gpu_ended_as(gpu: str) -> str:
    return f"::error::gpu-pr-gate ended as {gpu} (maintainer GPU smoke)"


def tier_summary(cpu: str, gpu: str) -> str:
    t1 = "ran" if cpu == "success" else "failed"
    if gpu == "success":
        smoke = "ran"
    elif gpu == "skipped":
        smoke = "skipped"
    else:
        smoke = "failed"
    # PR/push never runs gpu-nightly-full, so T2 stays NOT RUN.
    return ci_print_tier_summary.render(
        "unknown", t1, "not_run", gpu_smoke=smoke
    )


def evaluate(cpu: str, gpu: str) -> int:
    """Return the inference-paths exit code. cpu is judged before skipped."""
    sys.stdout.write(tier_summary(cpu, gpu))
    if cpu != "success":
        print(CPU_MUST_SUCCEED, file=sys.stderr)
        return 1
    if gpu == "skipped":
        print(NOT_A_TRT_PROOF)
        print(NOT_A_PROOF_SENTENCE)
        return 2
    if gpu == "success":
        print(TRT_PROOF_YES)
        return 0
    print(gpu_ended_as(gpu), file=sys.stderr)
    return 1


def extract_job(text: str, name: str) -> str:
    header = f"  {name}:"
    lines = text.splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        if line.startswith(header):
            start = i
            break
    if start is None:
        return ""
    body = [lines[start]]
    job_key = re.compile(r"^  [A-Za-z0-9-]+:")
    for line in lines[start + 1 :]:
        if job_key.match(line) and not line.startswith(header):
            break
        body.append(line)
    return "".join(body)


def check_workflow(workflow_path: Path | None = None) -> int:
    """Assert ci.yml wires needs results into this script. Shared by --self-test."""
    path = workflow_path or WORKFLOW
    if not path.is_file():
        print(f"[FAIL] workflow not found: {path}", file=sys.stderr)
        return 1
    text = path.read_text(encoding="utf-8")
    inference = extract_job(text, "inference-paths")
    gpu_job = extract_job(text, "gpu-pr-gate")
    errors: list[str] = []
    if not inference:
        errors.append("inference-paths job not found")
    else:
        if "python3 scripts/ci_eval_inference_paths.py" not in inference:
            errors.append("inference-paths does not call scripts/ci_eval_inference_paths.py")
        cpu_arg = "--cpu '${{ needs.cpu-profile.result }}'"
        gpu_arg = "--gpu '${{ needs.gpu-pr-gate.result }}'"
        if cpu_arg not in inference:
            errors.append(f"inference-paths is missing {cpu_arg}")
        if gpu_arg not in inference:
            errors.append(f"inference-paths is missing {gpu_arg}")
        if "if: ${{ always()" not in inference:
            errors.append("inference-paths if does not contain always()")
        if "continue-on-error" in inference:
            errors.append("inference-paths must not set continue-on-error")
        if INLINE_SKIP_AS_SUCCESS.search(inference):
            errors.append("inference-paths still treats skipped and success as exit 0")
    if not gpu_job:
        errors.append("gpu-pr-gate job not found")
    else:
        if "vars.MORTRED_HAS_GPU_RUNNER == 'true'" not in gpu_job:
            errors.append("gpu-pr-gate if lost MORTRED_HAS_GPU_RUNNER == 'true'")
        if "runs-on: [self-hosted, X64, gpu]" not in gpu_job:
            errors.append("gpu-pr-gate runs-on changed")
    if errors:
        for msg in errors:
            print(f"[FAIL] {msg}", file=sys.stderr)
        return 1
    print("[ok] ci.yml inference-paths wiring")
    return 0


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}", file=sys.stderr)
    return 1


def self_test() -> int:
    import contextlib
    import io

    cases = [
        ("success", "skipped", 2),
        ("success", "success", 0),
        ("success", "failure", 1),
        ("success", "cancelled", 1),
        ("failure", "skipped", 1),
        ("failure", "success", 1),
        ("success", "neutral", 1),
    ]
    for cpu, gpu, expect in cases:
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            code = evaluate(cpu, gpu)
        stdout = out_buf.getvalue()
        stderr = err_buf.getvalue()
        if code != expect:
            return _fail(f"--cpu {cpu} --gpu {gpu} exited {code}, expected {expect}")
        if cpu != "success":
            if "cpu-profile must succeed" not in stderr:
                return _fail("cpu failure must mention cpu-profile must succeed")
            if NOT_A_TRT_PROOF in stdout or TRT_PROOF_YES in stdout:
                return _fail("cpu failure must be decided before the skipped proof path")
            continue
        if gpu == "skipped":
            for needle in (
                NOT_A_TRT_PROOF,
                "SKIPPED",
                "NOT RUN",
                NOT_A_PROOF_SENTENCE,
            ):
                if needle not in stdout:
                    return _fail(f"skipped stdout missing {needle}")
            if TRT_PROOF_YES in stdout:
                return _fail("skipped stdout must not contain TRT_PROOF=yes")
            continue
        if gpu == "success":
            if TRT_PROOF_YES not in stdout or "RAN" not in stdout:
                return _fail("success stdout missing TRT_PROOF=yes or RAN")
            if NOT_A_TRT_PROOF in stdout:
                return _fail("success stdout must not contain NOT_A_TRT_PROOF")
            continue
        if gpu in ("failure", "cancelled"):
            if code == 2:
                return _fail("failure/cancelled must not use exit 2")
            if "gpu-pr-gate ended as" not in stderr:
                return _fail("failure/cancelled stderr missing gpu-pr-gate ended as")
            continue
        if TRT_PROOF_YES in stdout or NOT_A_TRT_PROOF in stdout:
            return _fail(f"gpu={gpu} must not claim a TRT proof")
    # Same workflow-source assertions as --check-workflow.
    workflow_rc = check_workflow()
    if workflow_rc != 0:
        return workflow_rc
    print("[ok] ci_eval_inference_paths.py --self-test")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", default=None, help="needs.cpu-profile.result")
    parser.add_argument("--gpu", default=None, help="needs.gpu-pr-gate.result")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check-workflow", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.check_workflow:
        return check_workflow()
    if args.cpu is None or args.gpu is None:
        print("usage: --cpu RESULT --gpu RESULT", file=sys.stderr)
        return 1
    return evaluate(args.cpu, args.gpu)


if __name__ == "__main__":
    sys.exit(main())
