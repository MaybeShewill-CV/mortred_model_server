#!/usr/bin/env python3
"""Read table 5 and the H3 record files. Does not start trtexec."""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h3_lines  # noqa: E402

FP32_IDS = {
    "mobilenetv2",
    "resnet",
    "densenet",
    "bisenetv2",
    "attentive_gan",
    "realesrgan",
}
SHAPES = {
    "libface": "input:1x3x480x640",
    "centerface": "input.1:1x3x640x640",
    "enlightengan": "input_src:1x3x256x256,input_gray:1x1x256x256",
    "realesrgan": "input:1x256x256x3",
}
YOLO8 = {
    "yolov8n": "weights/object_detection/yolov8/yolov8n.engine.static_bs1.fp16",
    "yolov8s": "weights/object_detection/yolov8/yolov8s.engine.static_bs1.fp16",
    "yolov8l": "weights/object_detection/yolov8/yolov8l.engine.static_bs1.fp16",
    "yolov8x": "weights/object_detection/yolov8/yolov8x.engine.static_bs1.fp16",
}
HEADER = "| id | hardware | command | metric | date |"
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
METRIC_RE = re.compile(r"trtexec_rps=[0-9]+(?:\.[0-9]+)?\Z")
PREFIX_ANCHORS = (
    "## 机器",
    "name, memory.total [MiB], memory.free [MiB], driver_version, compute_cap",
    "2026-09-24",
    "## 表 1 — 身份 / 转换",
    "## 表 2 — trtexec profile",
    "## 表 3 — 标定（非发布 RPS）",
    "## 表 4 — 真实 HTTP（发布值）",
    "## Gaps（失败/非 HTTP，来自 convert JSON）",
    "- `msocrnet`: convert+profile only (`http=False`)",
)


def split_cells(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None
    return [cell.strip() for cell in stripped[1:-1].split("|")]


def hardware_problems(hardware: str) -> list[str]:
    problems = []
    banned = ("name, memory.total", "memory.total", "driver_version", "compute_cap", "N/A")
    for item in banned:
        if item in hardware:
            problems.append("hardware contains %s" % item)
    parts = [part.strip() for part in hardware.split(",")]
    if len(parts) != 3 or any(part == "" for part in parts):
        problems.append("hardware is not three non-empty comma fields")
        return problems
    if "MiB" not in parts[1] and "GiB" not in parts[1]:
        problems.append("memory field has no MiB or GiB")
    return problems


def spec_problems() -> list[str]:
    problems = []
    if len(h3_lines.SPECS) != 39:
        problems.append("spec count %s" % len(h3_lines.SPECS))
    ids = [cid for cid, _engine, _shapes in h3_lines.SPECS]
    if len(ids) != len(set(ids)):
        problems.append("duplicate spec id")
    fp32 = {cid for cid, engine, _shapes in h3_lines.SPECS if engine.endswith(".fp32")}
    if fp32 != FP32_IDS:
        problems.append("fp32 id set mismatch")
    shapes = {cid: shapes for cid, _engine, shapes in h3_lines.SPECS if shapes}
    if shapes != SHAPES:
        problems.append("shapes mismatch")
    engines = {cid: engine for cid, engine, _shapes in h3_lines.SPECS}
    for cid, engine in YOLO8.items():
        if engines.get(cid) != engine:
            problems.append("yolov8 engine mismatch %s" % cid)
    libface = engines.get("libface", "")
    if libface != "weights/object_detection/libfacedetection/face_detection_yunet_2026may.engine.static_bs1.fp16":
        problems.append("libface engine mismatch")
    if "libface_320" in ids or "libface_640" in ids:
        problems.append("spec uses a split libface id")
    return problems


def check_text(text: str) -> list[str]:
    problems = spec_problems()
    mark = h3_lines.MARK
    count = text.count(mark)
    if count != 1:
        problems.append("table 5 heading count %s" % count)
        return problems
    idx = text.find(mark)
    prefix = text[:idx]
    section = text[idx:]
    cursor = -1
    for anchor in PREFIX_ANCHORS:
        pos = prefix.find(anchor)
        if pos < 0:
            problems.append("prefix missing: %s" % anchor)
            continue
        if pos <= cursor:
            problems.append("prefix out of order: %s" % anchor)
        cursor = pos
    if "libface_320" not in prefix or "libface_640" not in prefix:
        problems.append("prefix lost libface_320 or libface_640")
    if "libface_320" in section or "libface_640" in section:
        problems.append("table 5 contains libface_320 or libface_640")
    lines = section.splitlines()
    if not lines or lines[0] != mark:
        problems.append("section does not start at the heading")
        return problems
    body = [line for line in lines[1:] if line.strip()]
    if len(body) < 2:
        problems.append("table 5 has no header")
        return problems
    if body[0].strip() != HEADER:
        problems.append("table 5 header mismatch")
    sep = split_cells(body[1])
    if sep is None or len(sep) != 5 or any(re.fullmatch(r"-+", cell) is None for cell in sep):
        problems.append("table 5 separator mismatch")
    data = body[2:]
    if len(data) != 39:
        problems.append("table 5 row count %s" % len(data))
    rows = []
    for line in data:
        cells = split_cells(line)
        if cells is None or len(cells) != 5:
            problems.append("row is not five cells: %s" % line)
            continue
        rows.append(cells)
    if len(rows) != 39:
        return problems
    hardware_values = []
    commands = []
    for (cid, engine, shapes), cells in zip(h3_lines.SPECS, rows):
        got_id, hardware, command, metric, date = cells
        if got_id != cid:
            problems.append("id order %s != %s" % (got_id, cid))
        problems.extend("%s %s" % (cid, item) for item in hardware_problems(hardware))
        hardware_values.append(hardware)
        commands.append(command)
        expected = h3_lines.build_command(engine, shapes)
        if command != expected:
            problems.append("%s command mismatch" % cid)
        if not command.startswith("3rd_party/bin/trtexec"):
            problems.append("%s command does not start with trtexec" % cid)
        tokens = command.split()
        for flag in (
            "--loadEngine=%s" % engine,
            "--warmUp=500",
            "--duration=15",
            "--avgRuns=100",
            "--useSpinWait",
        ):
            if flag not in tokens:
                problems.append("%s missing %s" % (cid, flag))
        if "|" in command or "logs/bench" in command or command.endswith(".cmd.txt"):
            problems.append("%s command is not executable argv" % cid)
        if engine.endswith(".fp16") and "--fp16" not in tokens:
            problems.append("%s missing --fp16" % cid)
        if engine.endswith(".fp32") and "--fp16" in command:
            problems.append("%s fp32 command contains --fp16" % cid)
        if shapes:
            if "--shapes=%s" % shapes not in tokens:
                problems.append("%s missing shapes" % cid)
        elif "--shapes" in command:
            problems.append("%s unexpected shapes" % cid)
        if METRIC_RE.fullmatch(metric) is None:
            problems.append("%s metric %s" % (cid, metric))
        if DATE_RE.fullmatch(date) is None:
            problems.append("%s date %s" % (cid, date))
        else:
            try:
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                problems.append("%s date %s" % (cid, date))
        record = h3_lines.RECORD_DIR / ("%s.txt" % cid)
        if not record.is_file():
            problems.append("missing record %s" % cid)
            continue
        rec_lines = record.read_text(encoding="utf-8").splitlines()
        if len(rec_lines) != 4:
            problems.append("%s record line count %s" % (cid, len(rec_lines)))
            continue
        if rec_lines[0] != hardware or rec_lines[1] != date or rec_lines[2] != command:
            problems.append("%s record does not match the row" % cid)
        number = h3_lines.throughput_number(rec_lines[3])
        if "Throughput:" not in rec_lines[3] or number is None or metric != "trtexec_rps=" + number:
            problems.append("%s throughput mismatch" % cid)
    if len(set(hardware_values)) != 1:
        problems.append("hardware strings differ")
    if len(commands) == 39 and len(set(commands)) != 39:
        problems.append("commands are not unique")
    return problems


def main() -> int:
    if not h3_lines.DOC.is_file():
        print("missing %s" % h3_lines.DOC, file=sys.stderr)
        return 1
    text = h3_lines.DOC.read_text(encoding="utf-8")
    problems = check_text(text)
    if problems:
        for item in problems:
            print(item, file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
