#!/usr/bin/env python3
"""Collect one live trtexec line per Set A catalog id and publish table 5.

Metrics come only from the stdout of the trtexec process started here.
This script does not read table 2 or logs/bench.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "benchmark-set-a.md"
RECORD_DIR = ROOT / "docs" / "benchmark-h3"
MARK = "## 表 5 — 一行基准"
NUMBER_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?\Z")

# id, engine relative to the repo root, shapes (empty means no --shapes).
SPECS: tuple[tuple[str, str, str], ...] = (
    ("mobilenetv2", "weights/classification/mobilenetv2/mobilenetv2.engine.static_bs1.fp32", ""),
    ("resnet", "weights/classification/resnet/resnet.engine.static_bs1.fp32", ""),
    ("densenet", "weights/classification/densenet/densenet.engine.static_bs1.fp32", ""),
    ("yolov5l", "weights/object_detection/yolov5/yolov5l.engine.static_bs1.fp16", ""),
    ("yolov6s", "weights/object_detection/yolov6/yolov6s.engine.static_bs1.fp16", ""),
    ("nanodet_1x5", "weights/object_detection/nanodet/nanodet_plus_m_1x5.engine.static_bs1.fp16", ""),
    ("nanodet_416", "weights/object_detection/nanodet/nanodet_plus_m_416.engine.static_bs1.fp16", ""),
    ("yolov7", "weights/object_detection/yolov7/yolov7.engine.static_bs1.fp16", ""),
    ("yolov7x", "weights/object_detection/yolov7/yolov7x.engine.static_bs1.fp16", ""),
    ("yolov8n", "weights/object_detection/yolov8/yolov8n.engine.static_bs1.fp16", ""),
    ("yolov8s", "weights/object_detection/yolov8/yolov8s.engine.static_bs1.fp16", ""),
    ("yolov8l", "weights/object_detection/yolov8/yolov8l.engine.static_bs1.fp16", ""),
    ("yolov8x", "weights/object_detection/yolov8/yolov8x.engine.static_bs1.fp16", ""),
    ("libface", "weights/object_detection/libfacedetection/face_detection_yunet_2026may.engine.static_bs1.fp16", "input:1x3x480x640"),
    ("centerface", "weights/object_detection/centerface_detection/centerface.engine.static_bs1.fp16", "input.1:1x3x640x640"),
    ("bisenetv2", "weights/scene_segmentation/bisenetv2/bisenetv2_cityscapes.engine.static_bs1.fp32", ""),
    ("pphuman_mobile", "weights/scene_segmentation/pphuman_seg/pp_humanseg_192x192_mobile.engine.static_bs1.fp16", ""),
    ("pphuman_lite", "weights/scene_segmentation/pphuman_seg/pp_humanseg_192x192_lite.engine.static_bs1.fp16", ""),
    ("pphuman_server", "weights/scene_segmentation/pphuman_seg/pp_humanseg_512x512_server.engine.static_bs1.fp16", ""),
    ("hrnet", "weights/scene_segmentation/hrnet/hrnetw48_ccd.engine.static_bs1.fp16", ""),
    ("msocrnet", "weights/scene_segmentation/msocrnet/msocrnet_fp16.engine.static_bs1.fp16", ""),
    ("dbnet", "weights/ocr/db_text_detector/db_model_large.engine.static_bs1.fp16", ""),
    ("attentive_gan", "weights/enhancement/attentive_gan_derain/attentive_gan_derain.engine.static_bs1.fp32", ""),
    ("enlightengan", "weights/enhancement/enlighten_gan/enlightengan.engine.static_bs1.fp16", "input_src:1x3x256x256,input_gray:1x1x256x256"),
    ("realesrgan", "weights/enhancement/real_esrgan/realesr-general-x4v3.engine.static_bs1.fp32", "input:1x256x256x3"),
    ("superpoint", "weights/feature_point/superpoint/superpoint_120x160.engine.static_bs1.fp16", ""),
    ("dinov2_vits14", "weights/classification/dinov2/dinov2_vits14_pretrain.engine.static_bs1.fp16", ""),
    ("dinov2_vitb14", "weights/classification/dinov2/dinov2_vitb14_pretrain.engine.static_bs1.fp16", ""),
    ("dinov2_vitl14", "weights/classification/dinov2/dinov2_vitl14_pretrain.engine.static_bs1.fp16", ""),
    ("modnet", "weights/matting/modnet/modnet_hrnet_w18.engine.static_bs1.fp16", ""),
    ("ppmatting_512", "weights/matting/ppmatting/ppmatting_hrnet_w18_human_512.engine.static_bs1.fp16", ""),
    ("ppmatting_1024", "weights/matting/ppmatting/ppmatting_hrnet_w18_human_1024.engine.static_bs1.fp16", ""),
    ("ppmatting_resnet34", "weights/matting/ppmatting/pp_humanmatting-resnet34_vd.engine.static_bs1.fp16", ""),
    ("ppmatting_v2", "weights/matting/ppmatting/ppmattingv2_stdc1_human_512.engine.static_bs1.fp16", ""),
    ("depth_vits14", "weights/mono_depth_estimation/depth_anything/depth_anything_vits14.engine.static_bs1.fp16", ""),
    ("depth_vitb14", "weights/mono_depth_estimation/depth_anything/depth_anything_vitb14.engine.static_bs1.fp16", ""),
    ("depth_vitl14", "weights/mono_depth_estimation/depth_anything/depth_anything_vitl14.engine.static_bs1.fp16", ""),
    ("metric3d_512", "weights/mono_depth_estimation/metric3d/metric3d_750k_512x1088.engine.static_bs1.fp16", ""),
    ("metric3d_1088", "weights/mono_depth_estimation/metric3d/metric3d_750k_1088x1920.engine.static_bs1.fp16", ""),
)


def build_argv(engine: str, shapes: str) -> list[str]:
    argv = [
        "3rd_party/bin/trtexec",
        "--loadEngine=%s" % engine,
        "--warmUp=500",
        "--duration=15",
        "--avgRuns=100",
        "--useSpinWait",
    ]
    if engine.endswith(".fp16"):
        argv.append("--fp16")
    if shapes:
        argv.append("--shapes=%s" % shapes)
    return argv


def build_command(engine: str, shapes: str) -> str:
    return " ".join(build_argv(engine, shapes))


def hardware_ok(hardware: str) -> bool:
    banned = ("name, memory.total", "memory.total", "driver_version", "compute_cap", "N/A")
    if any(item in hardware for item in banned):
        return False
    parts = [part.strip() for part in hardware.split(",")]
    if len(parts) != 3 or any(part == "" for part in parts):
        return False
    if "MiB" not in parts[1] and "GiB" not in parts[1]:
        return False
    return True


def throughput_number(line: str) -> str | None:
    """Return the last Throughput/qps number on this line, unformatted."""
    last = None
    start = 0
    marker = "Throughput:"
    while True:
        idx = line.find(marker, start)
        if idx < 0:
            break
        rest = line[idx + len(marker) :]
        qps = rest.find("qps")
        if qps >= 0:
            token = rest[:qps].strip()
            if NUMBER_RE.fullmatch(token):
                last = token
        start = idx + len(marker)
    return last


def last_throughput(stdout: str) -> tuple[str, str] | None:
    found: tuple[str, str] | None = None
    for line in stdout.splitlines():
        number = throughput_number(line)
        if number is not None:
            found = (number, line)
    return found


def trt_env() -> dict[str, str]:
    env = os.environ.copy()
    extra = ["/usr/lib/wsl/lib", str(ROOT / "3rd_party" / "libs")]
    env["LD_LIBRARY_PATH"] = ":".join(extra + [env.get("LD_LIBRARY_PATH", "")])
    return env


def probe_hardware() -> str | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        print("FAIL nvidia-smi %s" % exc, file=sys.stderr)
        return None
    if proc.returncode != 0:
        print(proc.stderr or "", file=sys.stderr)
        return None
    lines = (proc.stdout or "").splitlines()
    if not lines:
        return None
    hardware = lines[0].strip()
    if not hardware_ok(hardware):
        print("FAIL hardware %s" % hardware, file=sys.stderr)
        return None
    return hardware


def render(rows: list[dict]) -> str:
    lines = [
        MARK,
        "",
        "| id | hardware | command | metric | date |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (row["id"], row["hardware"], row["command"], row["metric"], row["date"])
        )
    return "\n".join(lines) + "\n"


def publish(rows: list[dict]) -> None:
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    for row in rows:
        payload = "%s\n%s\n%s\n%s\n" % (
            row["hardware"],
            row["date"],
            row["command"],
            row["throughput_line"],
        )
        path = RECORD_DIR / ("%s.txt" % row["id"])
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(payload)
    raw = DOC.read_bytes()
    marker = MARK.encode("utf-8")
    idx = raw.find(marker)
    prefix = raw[:idx] if idx >= 0 else raw
    nl = b"\r\n" if b"\r\n" in prefix else b"\n"
    if prefix and not prefix.endswith(nl):
        prefix += nl
    if not prefix.endswith(nl * 2):
        prefix += nl
    section = render(rows).encode("utf-8")
    if nl == b"\r\n":
        section = section.replace(b"\n", b"\r\n")
    DOC.write_bytes(prefix + section)


def main() -> int:
    if len(SPECS) != 39:
        print("FAIL spec count %s" % len(SPECS), file=sys.stderr)
        return 1
    hardware = probe_hardware()
    if hardware is None:
        for cid, _engine, _shapes in SPECS:
            print("FAIL %s" % cid, file=sys.stderr)
        return 1
    print("hardware %s" % hardware, flush=True)
    done: list[dict] = []
    failed: list[str] = []
    for index, (cid, engine, shapes) in enumerate(SPECS, start=1):
        command = build_command(engine, shapes)
        if not (ROOT / engine).is_file():
            print("FAIL %s engine missing" % cid, file=sys.stderr, flush=True)
            failed.append(cid)
            continue
        print("run %s/%s %s" % (index, len(SPECS), cid), flush=True)
        try:
            proc = subprocess.run(
                build_argv(engine, shapes),
                cwd=str(ROOT),
                env=trt_env(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            print("FAIL %s %s" % (cid, exc), file=sys.stderr, flush=True)
            failed.append(cid)
            continue
        date = time.strftime("%Y-%m-%d")
        parsed = last_throughput(proc.stdout or "")
        if parsed is not None:
            print(parsed[1], flush=True)
        if proc.returncode != 0 or parsed is None:
            print("FAIL %s rc=%s" % (cid, proc.returncode), file=sys.stderr, flush=True)
            failed.append(cid)
            continue
        number, line = parsed
        done.append(
            {
                "id": cid,
                "hardware": hardware,
                "command": command,
                "metric": "trtexec_rps=%s" % number,
                "date": date,
                "throughput_line": line,
            }
        )
    if failed or len(done) != len(SPECS):
        print("not published", file=sys.stderr)
        for cid in failed:
            print(cid, file=sys.stderr)
        return 1
    publish(done)
    print("wrote %s" % DOC.relative_to(ROOT), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
