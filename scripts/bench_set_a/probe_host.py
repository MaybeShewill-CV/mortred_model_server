#!/usr/bin/env python3
"""Collect host fingerprint and A-set static_bs1 ONNX inventory."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_PARTS = {"diffusion", "openai_clip", "sam", "lightglue"}


def run(cmd: list[str], timeout: int = 30) -> str:
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
        return (p.stdout or "") + (p.stderr or "")
    except Exception as exc:
        return "ERR %s" % exc


def main() -> int:
    extra = []
    wsl_lib = Path("/usr/lib/wsl/lib")
    lib_dir = ROOT / "3rd_party" / "libs"
    if wsl_lib.is_dir():
        extra.append(str(wsl_lib))
    if lib_dir.is_dir():
        extra.append(str(lib_dir))
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(extra + [env.get("LD_LIBRARY_PATH", "")])

    smi = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
            "--format=csv",
        ]
    )
    trtexec = ROOT / "3rd_party" / "bin" / "trtexec"
    trt_out = ""
    if trtexec.is_file():
        p = subprocess.run(
            [str(trtexec)],
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
        )
        trt_out = ((p.stdout or "") + (p.stderr or ""))[:4000]

    server = ROOT / "_bin" / "mortred-model-server.out"
    asan = False
    nvinfer = False
    if server.is_file():
        ldd = run(["ldd", str(server)])
        asan = "libasan" in ldd or "libclang_rt.asan" in ldd
        nvinfer = "libnvinfer" in ldd

    files = []
    weights = ROOT / "weights"
    if weights.is_dir():
        for p in sorted(weights.rglob("*.static_bs1.onnx")):
            rel = p.relative_to(ROOT).as_posix()
            parts = set(rel.split("/"))
            if parts & SKIP_PARTS:
                continue
            files.append(
                {
                    "onnx": rel,
                    "bytes": p.stat().st_size,
                }
            )

    engines = []
    if weights.is_dir():
        for p in sorted(weights.rglob("*.engine*")):
            rel = p.relative_to(ROOT).as_posix()
            if set(rel.split("/")) & SKIP_PARTS:
                continue
            engines.append(rel)

    host = {
        "uname": run(["uname", "-a"]).strip(),
        "nvidia_smi": smi.strip(),
        "trtexec_path": str(trtexec) if trtexec.is_file() else None,
        "trtexec_banner": trt_out.strip(),
        "server": str(server) if server.is_file() else None,
        "server_asan": asan,
        "server_nvinfer": nvinfer,
        "a_set_count": len(files),
        "a_set_onnx": files,
        "existing_engines": engines,
    }
    out_dir = ROOT / "logs" / "bench" / "set_a"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "host.json").write_text(json.dumps(host, indent=2), encoding="utf-8")
    print("a_set_count", len(files))
    print("server", host["server"], "asan", asan, "nvinfer", nvinfer)
    print("trtexec", host["trtexec_path"])
    print(smi)
    for row in files:
        print(row["onnx"], row["bytes"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
