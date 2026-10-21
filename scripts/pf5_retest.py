#!/usr/bin/env python3
"""PF5 same-hardware retest for recorded HTTP RPS baselines.

Compares a current metrics JSON (http_infer_rps.py --out, or the same shape as
docs/perf/baselines/<id>.json) to the recorded baseline.

Thresholds from the eval standard:
  success RPS >= 85% of the recorded value (RPS_FLOOR)
  p99 <= 150% of the recorded value (P99_CEIL)

A hardware mismatch fails and does not rewrite the baseline. An intentional
RPS/p99 regression passes only when a changelog entry or --reason-file gives a
non-empty reason for that id. Command mismatches (worker_nums, concurrency,
load-command flags) are not excused by the changelog.

--ids all is a stub for the full product catalog (P1-FC-1). It does not invent
RPS numbers.

This script does not start a server and does not measure GPU RPS.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shlex
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINES = ROOT / "docs" / "perf" / "baselines"
DEFAULT_MARKDOWN = ROOT / "docs" / "perf" / "coverage_set_http_rps.md"
DEFAULT_CHANGELOG_NAME = "regressions.json"
PRODUCT_INDEX = ROOT / "conf" / "product_static_bs1.json"

RPS_FLOOR = 0.85
P99_CEIL = 1.50
MIN_WALL_S = 30.0
SCHEMA = "pf5-baseline-v1"
CHANGELOG_SCHEMA = "pf5-regression-v1"

COVERAGE_SET_IDS = (
    "mobilenetv2",
    "yolov8s",
    "libface",
    "pphuman_mobile",
    "dbnet",
    "dinov2_vits14",
    "superpoint",
    "realesrgan",
    "metric3d_512",
)

# Tokens copied from docs/perf/coverage_set_http_rps.md. Self-test uses these
# to lock the importer to that report. They are not new measurements.
RECORDED_TOKENS = {
    "mobilenetv2": ("894.751", "14.556", "4", "2026-10-14", "cpu"),
    "yolov8s": ("272.321", "42.064", "4", "2026-10-14", "cpu"),
    "libface": ("389.507", "28.819", "4", "2026-10-14", "gpu"),
    "pphuman_mobile": ("374.841", "12.521", "2", "2026-10-14", "gpu"),
    "dbnet": ("168.67", "25.311", "2", "2026-10-14", "gpu"),
    "dinov2_vits14": ("179.725", "23.838", "2", "2026-10-14", "cpu"),
    "superpoint": ("206.978", "17.245", "1", "2026-10-15", "gpu"),
    "realesrgan": ("24.568", "383.273", "4", "2026-10-15", "cpu"),
    "metric3d_512": ("22.514", "126.457", "1", "2026-10-15", "gpu"),
}
RECORDED_HARDWARE = "NVIDIA GeForce RTX 2070 SUPER, 8192 MiB"

BASELINE_KEYS = (
    "schema",
    "id",
    "source",
    "date",
    "hardware",
    "worker_nums",
    "decode_auto",
    "success_rps",
    "p50_ms",
    "p99_ms",
    "concurrency",
    "duration",
    "warmup",
    "warmup_s",
    "wall_s",
    "error_rate",
    "load_command",
    "load_url",
    "image",
    "server_uri",
    "conf_server",
    "engine",
)

SECTION_RE = re.compile(r"(?m)^## ([A-Za-z0-9_]+)\s*$")
ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
INFER_PATH_RE = re.compile(r"(/v1/models/([^/\s]+)/infer)\b")


class Pf5Error(Exception):
    """A caller-facing PF5 failure that should exit 1."""


class BaselineHardwareError(Pf5Error):
    """Existing baseline hardware does not match. The file is left untouched."""


def fmt_num(value: float) -> str:
    text = "%.6f" % float(value)
    return text.rstrip("0").rstrip(".")


def threshold_text(value: float) -> str:
    return "%.2f" % value


def parse_duration(text: str) -> float:
    """'30', '30s', '2m', '1h', '500ms' -> seconds. Same units as http_infer_rps."""
    raw = text.strip().lower()
    if not raw:
        raise Pf5Error("empty duration")
    unit = 1.0
    if raw.endswith("ms"):
        unit = 0.001
        raw = raw[:-2]
    elif raw[-1] in "smh":
        unit = {"s": 1.0, "m": 60.0, "h": 3600.0}[raw[-1]]
        raw = raw[:-1]
    try:
        value = float(raw)
    except ValueError as exc:
        raise Pf5Error("invalid duration %r" % text) from exc
    if value < 0:
        raise Pf5Error("duration must be non-negative")
    return value * unit


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Pf5Error("%s must be a number, got %r" % (name, value))
    return float(value)


def _positive(value, name: str) -> float:
    number = _number(value, name)
    if number <= 0:
        raise Pf5Error("%s must be positive, got %s" % (name, fmt_num(number)))
    return number


def _optional_number(value, name: str):
    if value is None:
        return None
    return _number(value, name)


def _int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Pf5Error("%s must be an integer, got %r" % (name, value))
    if value <= 0:
        raise Pf5Error("%s must be positive, got %s" % (name, value))
    return value


def check_model_id(model_id: str) -> str:
    if not isinstance(model_id, str) or not ID_RE.match(model_id):
        raise Pf5Error("invalid model id %r" % model_id)
    return model_id


def parse_kv_line(line: str) -> dict[str, str]:
    """Parse `k=v | k=\"quoted\"` fields from one coverage-report row."""
    fields: dict[str, str] = {}
    i = 0
    n = len(line)
    while i < n:
        while i < n and line[i] in " \t|":
            i += 1
        if i >= n:
            break
        key_start = i
        while i < n and line[i] not in "= \t|":
            i += 1
        key = line[key_start:i]
        if not key or i >= n or line[i] != "=":
            raise Pf5Error("expected key=value near %r" % line[key_start : key_start + 40])
        i += 1
        if i < n and line[i] == '"':
            i += 1
            chars: list[str] = []
            closed = False
            while i < n:
                ch = line[i]
                if ch == "\\":
                    if i + 1 >= n:
                        raise Pf5Error("trailing escape in %s" % key)
                    chars.append(line[i + 1])
                    i += 2
                    continue
                if ch == '"':
                    i += 1
                    closed = True
                    break
                chars.append(ch)
                i += 1
            if not closed:
                raise Pf5Error("unterminated quote in %s" % key)
            value = "".join(chars)
        else:
            start = i
            while i < n and line[i] != "|":
                i += 1
            value = line[start:i].strip()
        fields[key] = value
    return fields


def parse_coverage_markdown(text: str) -> list[dict[str, str]]:
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        raise Pf5Error("no ## id sections in coverage markdown")
    rows: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        lines = [item for item in body.splitlines() if item.strip()]
        model_id = check_model_id(match.group(1))
        if not lines or lines[0].strip() == "checked=no":
            continue
        fields = parse_kv_line(lines[0])
        if fields.get("checked") not in (None, "yes"):
            raise Pf5Error("%s checked=%r" % (model_id, fields.get("checked")))
        fields["id"] = model_id
        rows.append(fields)
    return rows


def _flag(command: str, *names: str) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise Pf5Error("load_command is not a shell command: %s" % exc) from exc
    for name in names:
        if name in tokens:
            index = tokens.index(name)
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
                return tokens[index + 1]
        prefix = name + "="
        for token in tokens:
            if token.startswith(prefix):
                return token[len(prefix) :]
    return None


def command_signature(command: str) -> dict[str, str | None]:
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise Pf5Error("load_command is not a shell command: %s" % exc) from exc
    return {
        "raw": "yes" if "--raw" in tokens else "no",
        "model": _flag(command, "--model"),
        "concurrency": _flag(command, "-c", "--concurrency"),
        "duration": _flag(command, "-d", "--duration"),
        "warmup": _flag(command, "--warmup"),
        "image": _flag(command, "--image"),
        "gateway": _flag(command, "--gateway"),
    }


def _blank_image(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "-"):
        return None
    return text


def build_baseline(
    *,
    model_id: str,
    source: str,
    date: str,
    hardware: str,
    worker_nums: int,
    decode_auto: str | None,
    success_rps: float,
    p50_ms: float | None,
    p99_ms: float,
    load_command: str,
    load_url: str | None,
    image: str | None,
    server_uri: str | None,
    conf_server: str | None,
    engine: str | None,
    wall_s: float | None,
    error_rate: float | None,
    concurrency: int | None = None,
    duration: str | None = None,
    warmup: str | None = None,
) -> dict:
    check_model_id(model_id)
    if not DATE_RE.match(str(date)):
        raise Pf5Error("%s date must be YYYY-MM-DD, got %r" % (model_id, date))
    hardware = str(hardware).strip()
    if not hardware:
        raise Pf5Error("%s hardware is empty" % model_id)
    if not str(load_command).strip():
        raise Pf5Error("%s load_command is empty" % model_id)
    if concurrency is None:
        parsed = _flag(load_command, "-c", "--concurrency")
        concurrency = int(parsed) if parsed is not None else None
    if duration is None:
        duration = _flag(load_command, "-d", "--duration")
    if warmup is None:
        warmup = _flag(load_command, "--warmup")
    warmup_s = parse_duration(warmup) if warmup else None
    success_rps = _positive(success_rps, "%s success_rps" % model_id)
    p99_ms = _positive(p99_ms, "%s p99_ms" % model_id)
    worker_nums = _int(worker_nums, "%s worker_nums" % model_id)
    if concurrency is not None:
        concurrency = _int(concurrency, "%s concurrency" % model_id)
        if concurrency != worker_nums * 2:
            raise Pf5Error(
                "%s concurrency %s != worker_nums*2 (%s)"
                % (model_id, concurrency, worker_nums * 2)
            )
    doc = {
        "schema": SCHEMA,
        "id": model_id,
        "source": source,
        "date": str(date),
        "hardware": hardware,
        "worker_nums": worker_nums,
        "decode_auto": None if decode_auto is None else str(decode_auto),
        "success_rps": success_rps,
        "p50_ms": None if p50_ms is None else _positive(p50_ms, "%s p50_ms" % model_id),
        "p99_ms": p99_ms,
        "concurrency": concurrency,
        "duration": duration,
        "warmup": warmup,
        "warmup_s": warmup_s,
        "wall_s": None if wall_s is None else _number(wall_s, "%s wall_s" % model_id),
        "error_rate": None if error_rate is None else _number(error_rate, "%s error_rate" % model_id),
        "load_command": str(load_command),
        "load_url": load_url,
        "image": _blank_image(image),
        "server_uri": server_uri,
        "conf_server": conf_server,
        "engine": engine,
    }
    return {key: doc[key] for key in BASELINE_KEYS}


def baseline_from_markdown_fields(fields: dict[str, str]) -> dict:
    model_id = fields["id"]
    required = ("success_rps", "p99_ms", "hardware", "worker_nums", "date", "load_command")
    missing = [key for key in required if not str(fields.get(key) or "").strip()]
    if missing:
        raise Pf5Error("%s markdown row missing %s" % (model_id, ", ".join(missing)))
    p50 = fields.get("p50_ms")
    wall = fields.get("wall_s")
    error_rate = fields.get("error_rate")
    return build_baseline(
        model_id=model_id,
        source="docs/perf/coverage_set_http_rps.md",
        date=fields["date"],
        hardware=fields["hardware"],
        worker_nums=int(fields["worker_nums"]),
        decode_auto=fields.get("decode_auto"),
        success_rps=float(fields["success_rps"]),
        p50_ms=None if p50 in (None, "") else float(p50),
        p99_ms=float(fields["p99_ms"]),
        load_command=fields["load_command"],
        load_url=fields.get("load_url"),
        image=fields.get("image"),
        server_uri=fields.get("server_uri"),
        conf_server=fields.get("conf_server"),
        engine=fields.get("engine"),
        wall_s=None if wall in (None, "") else float(wall),
        error_rate=None if error_rate in (None, "") else float(error_rate),
    )


def baseline_from_report_row(row: dict) -> dict:
    """Map a coverage_http_report.py row dict onto the baseline schema."""
    if row.get("blank"):
        raise Pf5Error("refusing a blank coverage row for %s" % row.get("id"))
    model_id = str(row.get("id") or "")
    return build_baseline(
        model_id=model_id,
        source="coverage_http_report.py",
        date=str(row.get("date") or ""),
        hardware=str(row.get("hardware") or ""),
        worker_nums=int(row["worker_nums"]),
        decode_auto=None if row.get("decode_auto") is None else str(row.get("decode_auto")),
        success_rps=float(row["success_rps"]),
        p50_ms=None if row.get("p50_ms") is None else float(row["p50_ms"]),
        p99_ms=float(row["p99_ms"]),
        load_command=str(row.get("load_command") or ""),
        load_url=None if not row.get("load_url") else str(row.get("load_url")),
        image=row.get("image"),
        server_uri=None if not row.get("server_uri") else str(row.get("server_uri")),
        conf_server=None if not row.get("conf_server") else str(row.get("conf_server")),
        engine=None if not row.get("engine") else str(row.get("engine")),
        wall_s=None if row.get("wall_s") is None else float(row["wall_s"]),
        error_rate=None if row.get("error_rate") is None else float(row["error_rate"]),
    )


def dump_baseline(doc: dict) -> str:
    ordered = {key: doc[key] for key in BASELINE_KEYS}
    return json.dumps(ordered, indent=2, ensure_ascii=False) + "\n"


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def baseline_path(baselines_dir: Path, model_id: str) -> Path:
    check_model_id(model_id)
    return baselines_dir / ("%s.json" % model_id)


def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Pf5Error("cannot read %s: %s" % (path, exc)) from exc
    if not isinstance(data, dict):
        raise Pf5Error("%s must be a JSON object" % path)
    return data


def validate_baseline(doc: dict, path: Path | None = None) -> dict:
    where = str(path) if path is not None else doc.get("id")
    if doc.get("schema") != SCHEMA:
        raise Pf5Error("%s schema must be %s" % (where, SCHEMA))
    model_id = check_model_id(str(doc.get("id") or ""))
    if path is not None and path.name != "%s.json" % model_id:
        raise Pf5Error("%s id %s does not match filename" % (path, model_id))
    hardware = str(doc.get("hardware") or "").strip()
    if not hardware:
        raise Pf5Error("%s hardware is empty" % model_id)
    if not DATE_RE.match(str(doc.get("date") or "")):
        raise Pf5Error("%s date must be YYYY-MM-DD" % model_id)
    _positive(doc.get("success_rps"), "%s success_rps" % model_id)
    _positive(doc.get("p99_ms"), "%s p99_ms" % model_id)
    worker_nums = _int(doc.get("worker_nums"), "%s worker_nums" % model_id)
    load_command = str(doc.get("load_command") or "")
    if not load_command.strip():
        raise Pf5Error("%s load_command is empty" % model_id)
    if "--raw" not in shlex.split(load_command):
        raise Pf5Error("%s load_command is not --raw" % model_id)
    if _flag(load_command, "--model") != model_id:
        raise Pf5Error("%s load_command --model does not match id" % model_id)
    concurrency = doc.get("concurrency")
    if concurrency is not None:
        concurrency = _int(concurrency, "%s concurrency" % model_id)
        if concurrency != worker_nums * 2:
            raise Pf5Error(
                "%s concurrency %s != worker_nums*2 (%s)"
                % (model_id, concurrency, worker_nums * 2)
            )
    missing = [key for key in BASELINE_KEYS if key not in doc]
    if missing:
        raise Pf5Error("%s baseline missing keys: %s" % (model_id, ", ".join(missing)))
    return doc


def load_baseline(path: Path) -> dict:
    return validate_baseline(load_json(path), path)


def record_baseline(row: dict, baselines_dir: Path) -> str:
    """Create docs/perf/baselines/<id>.json the first time a PF1 row is produced.

    Same hardware and an existing file: keep the recorded numbers ("kept").
    Different hardware: raise BaselineHardwareError and leave the file bytes
    unchanged. This function never replaces success_rps or p99_ms in place;
    that is `pf5_retest.py --adopt-baseline` plus a changelog reason.
    """
    doc = baseline_from_report_row(row)
    path = baseline_path(baselines_dir, doc["id"])
    if not path.is_file():
        atomic_write(path, dump_baseline(doc))
        return "created"
    before = path.read_bytes()
    try:
        existing = load_baseline(path)
    except Pf5Error:
        path.write_bytes(before)
        raise
    if str(existing.get("hardware") or "").strip() != doc["hardware"]:
        raise BaselineHardwareError(
            "PF5 baseline hardware mismatch for %s: baseline=%r current=%r; "
            "refusing to rewrite %s"
            % (doc["id"], existing.get("hardware"), doc["hardware"], path)
        )
    if path.read_bytes() != before:
        path.write_bytes(before)
        raise Pf5Error("refusing to rewrite %s" % path)
    return "kept"


def adopt_baseline(current: dict, baselines_dir: Path, reason: str, hardware_flag: str | None) -> str:
    """Replace recorded RPS/p99 from a measured current JSON.

    Requires a non-empty reason and the same hardware. Refuses command-shape
    changes (worker_nums / concurrency / load-command flags).
    """
    text = reason.strip()
    if not text:
        raise Pf5Error("adopt-baseline requires a non-empty changelog reason")
    model_id = str(current.get("id") or "")
    if not model_id:
        url = str(current.get("url") or current.get("load_url") or "")
        found = INFER_PATH_RE.search(url)
        if found:
            model_id = found.group(2)
    model_id = check_model_id(model_id)
    path = baseline_path(baselines_dir, model_id)
    if not path.is_file():
        raise Pf5Error("no baseline to adopt for %s (%s)" % (model_id, path))
    before = path.read_bytes()
    existing = load_baseline(path)
    metrics = metrics_from_current(current, model_id, hardware_flag)
    if metrics["hardware"] != str(existing["hardware"]).strip():
        raise BaselineHardwareError(
            "FAIL %s hardware mismatch: baseline=%r current=%r; baseline not rewritten"
            % (model_id, existing["hardware"], metrics["hardware"])
        )
    problems = _command_problems(existing, metrics)
    if problems:
        raise Pf5Error(
            "FAIL %s refusing to adopt because the command does not match: %s"
            % (model_id, "; ".join(problems))
        )
    updated = dict(existing)
    updated["success_rps"] = metrics["success_rps"]
    updated["p99_ms"] = metrics["p99_ms"]
    if metrics.get("p50_ms") is not None:
        updated["p50_ms"] = metrics["p50_ms"]
    if metrics.get("wall_s") is not None:
        updated["wall_s"] = metrics["wall_s"]
    if metrics.get("error_rate") is not None:
        updated["error_rate"] = metrics["error_rate"]
    if current.get("date"):
        updated["date"] = str(current["date"])
    updated["source"] = "pf5_retest.py --adopt-baseline"
    updated["adopted_reason"] = text
    # adopted_reason is recorded beside the schema keys.
    ordered = {key: updated[key] for key in BASELINE_KEYS}
    ordered["adopted_reason"] = text
    try:
        atomic_write(path, json.dumps(ordered, indent=2, ensure_ascii=False) + "\n")
    except OSError:
        path.write_bytes(before)
        raise
    return (
        "adopted %s success_rps %s -> %s p99_ms %s -> %s reason: %s"
        % (
            model_id,
            fmt_num(float(existing["success_rps"])),
            fmt_num(metrics["success_rps"]),
            fmt_num(float(existing["p99_ms"])),
            fmt_num(metrics["p99_ms"]),
            text,
        )
    )


def metrics_from_current(doc: dict, model_id: str, hardware_flag: str | None) -> dict:
    """Normalize http_infer_rps --out JSON or a baseline-shaped document."""
    check_model_id(model_id)
    url = str(doc.get("url") or doc.get("load_url") or "")
    url_id = None
    found = INFER_PATH_RE.search(url)
    if found:
        url_id = found.group(2)
    doc_id = doc.get("id")
    if doc_id is not None and str(doc_id) != model_id:
        raise Pf5Error("current id %r does not match --ids %s" % (doc_id, model_id))
    if url_id is not None and url_id != model_id:
        raise Pf5Error("current url id %r does not match --ids %s" % (url_id, model_id))
    if "success_rps" in doc:
        success_rps = doc.get("success_rps")
    elif "rps" in doc:
        success_rps = doc.get("rps")
    else:
        raise Pf5Error("%s current JSON has no success_rps or rps" % model_id)
    if "p99_ms" in doc:
        p99_ms = doc.get("p99_ms")
    else:
        latency = doc.get("latency_ms")
        if not isinstance(latency, dict) or "p99" not in latency:
            raise Pf5Error("%s current JSON has no p99_ms or latency_ms.p99" % model_id)
        p99_ms = latency.get("p99")
    hardware = str(doc.get("hardware") or "").strip()
    if not hardware:
        hardware = (hardware_flag or "").strip()
    if not hardware:
        raise Pf5Error(
            "%s current metrics have no hardware; pass --hardware "
            "(refusing to compare against an unknown machine or rewrite the baseline)"
            % model_id
        )
    if hardware_flag and str(doc.get("hardware") or "").strip() and hardware_flag.strip() != hardware:
        raise Pf5Error(
            "%s --hardware %r does not match current JSON hardware %r"
            % (model_id, hardware_flag.strip(), hardware)
        )
    p50 = doc.get("p50_ms")
    if p50 is None and isinstance(doc.get("latency_ms"), dict):
        p50 = doc["latency_ms"].get("p50")
    worker_nums = doc.get("worker_nums")
    concurrency = doc.get("concurrency")
    load_command = doc.get("load_command")
    if load_command is not None:
        load_command = str(load_command)
        if concurrency is None:
            parsed = _flag(load_command, "-c", "--concurrency")
            concurrency = int(parsed) if parsed is not None else None
    if worker_nums is not None:
        worker_nums = _int(worker_nums, "%s worker_nums" % model_id)
    if concurrency is not None:
        concurrency = _int(concurrency, "%s concurrency" % model_id)
    evidenced = worker_nums is not None or concurrency is not None or bool(load_command)
    if not evidenced:
        raise Pf5Error(
            "%s current metrics omit worker_nums, concurrency, and load_command; "
            "cannot show this is the same command" % model_id
        )
    warmup_s = doc.get("warmup_s")
    if warmup_s is None and doc.get("warmup"):
        warmup_s = parse_duration(str(doc["warmup"]))
    duration_s = None
    if doc.get("duration"):
        duration_s = parse_duration(str(doc["duration"]))
    return {
        "id": model_id,
        "success_rps": _positive(success_rps, "%s success_rps" % model_id),
        "p99_ms": _positive(p99_ms, "%s p99_ms" % model_id),
        "p50_ms": None if p50 is None else _number(p50, "%s p50_ms" % model_id),
        "hardware": hardware,
        "worker_nums": worker_nums,
        "concurrency": concurrency,
        "decode_auto": None if doc.get("decode_auto") is None else str(doc.get("decode_auto")),
        "load_command": load_command,
        "load_url": url or None,
        "warmup_s": None if warmup_s is None else _number(warmup_s, "%s warmup_s" % model_id),
        "wall_s": None if doc.get("wall_s") is None else _number(doc.get("wall_s"), "%s wall_s" % model_id),
        "error_rate": None
        if doc.get("error_rate") is None
        else _number(doc.get("error_rate"), "%s error_rate" % model_id),
        "duration_s": duration_s,
    }


def _command_problems(baseline: dict, current: dict) -> list[str]:
    problems: list[str] = []
    if current.get("worker_nums") is not None and int(current["worker_nums"]) != int(baseline["worker_nums"]):
        problems.append(
            "worker_nums baseline=%s current=%s" % (baseline["worker_nums"], current["worker_nums"])
        )
    base_conc = baseline.get("concurrency")
    if current.get("concurrency") is not None and base_conc is not None:
        if int(current["concurrency"]) != int(base_conc):
            problems.append(
                "concurrency baseline=%s current=%s" % (base_conc, current["concurrency"])
            )
    if (
        current.get("decode_auto") is not None
        and baseline.get("decode_auto") is not None
        and str(current["decode_auto"]) != str(baseline["decode_auto"])
    ):
        problems.append(
            "decode_auto baseline=%s current=%s" % (baseline["decode_auto"], current["decode_auto"])
        )
    if current.get("load_command") and baseline.get("load_command"):
        left = command_signature(str(baseline["load_command"]))
        right = command_signature(str(current["load_command"]))
        if left != right:
            problems.append("load_command flags baseline=%s current=%s" % (left, right))
    base_url = str(baseline.get("load_url") or "")
    cur_url = str(current.get("load_url") or "")
    if base_url and cur_url:
        base_path = INFER_PATH_RE.search(base_url)
        cur_path = INFER_PATH_RE.search(cur_url)
        if base_path and cur_path and base_path.group(1) != cur_path.group(1):
            problems.append("infer path baseline=%s current=%s" % (base_path.group(1), cur_path.group(1)))
    if current.get("warmup_s") is not None and baseline.get("warmup_s") is not None:
        if abs(float(current["warmup_s"]) - float(baseline["warmup_s"])) > 1e-3:
            problems.append(
                "warmup_s baseline=%s current=%s" % (baseline["warmup_s"], current["warmup_s"])
            )
    if current.get("duration_s") is not None and baseline.get("duration"):
        expected = parse_duration(str(baseline["duration"]))
        if abs(float(current["duration_s"]) - expected) > 1e-3:
            problems.append("duration baseline=%s current=%s" % (baseline["duration"], current["duration_s"]))
    if current.get("wall_s") is not None and float(current["wall_s"]) < MIN_WALL_S:
        problems.append("wall_s %s < %s" % (fmt_num(float(current["wall_s"])), fmt_num(MIN_WALL_S)))
    return problems


def compare_metrics(baseline: dict, current: dict, reason: str | None) -> tuple[bool, str]:
    """Return (ok, line). Hardware and command failures ignore `reason`."""
    model_id = str(baseline["id"])
    base_hw = str(baseline.get("hardware") or "").strip()
    cur_hw = str(current.get("hardware") or "").strip()
    if not cur_hw:
        return False, (
            "FAIL %s current metrics have no hardware; pass --hardware "
            "(baseline not rewritten)" % model_id
        )
    if base_hw != cur_hw:
        return False, (
            "FAIL %s hardware mismatch: baseline=%r current=%r (baseline not rewritten)"
            % (model_id, base_hw, cur_hw)
        )
    problems = _command_problems(baseline, current)
    if problems:
        return False, "FAIL %s same-command check failed: %s (baseline not rewritten)" % (
            model_id,
            "; ".join(problems),
        )
    base_rps = float(baseline["success_rps"])
    base_p99 = float(baseline["p99_ms"])
    cur_rps = float(current["success_rps"])
    cur_p99 = float(current["p99_ms"])
    rps_ratio = cur_rps / base_rps
    p99_ratio = cur_p99 / base_p99
    misses = []
    if rps_ratio < RPS_FLOOR:
        misses.append(
            "success_rps %s -> %s (ratio %s < %s)"
            % (fmt_num(base_rps), fmt_num(cur_rps), fmt_num(rps_ratio), threshold_text(RPS_FLOOR))
        )
    if p99_ratio > P99_CEIL:
        misses.append(
            "p99_ms %s -> %s (ratio %s > %s)"
            % (fmt_num(base_p99), fmt_num(cur_p99), fmt_num(p99_ratio), threshold_text(P99_CEIL))
        )
    if misses and reason and reason.strip():
        return True, "PASS %s intentional regression: %s; reason: %s" % (
            model_id,
            "; ".join(misses),
            reason.strip(),
        )
    if misses:
        return False, "FAIL %s %s" % (model_id, "; ".join(misses))
    return True, (
        "PASS %s success_rps %s -> %s (ratio %s >= %s); p99_ms %s -> %s (ratio %s <= %s)"
        % (
            model_id,
            fmt_num(base_rps),
            fmt_num(cur_rps),
            fmt_num(rps_ratio),
            threshold_text(RPS_FLOOR),
            fmt_num(base_p99),
            fmt_num(cur_p99),
            fmt_num(p99_ratio),
            threshold_text(P99_CEIL),
        )
    )


def resolve_ids(spec: str) -> list[str] | None:
    """Return ids to compare, or None when `spec` is the full-catalog stub."""
    text = spec.strip()
    if text == "all":
        return None
    if text == "coverage_set":
        return list(COVERAGE_SET_IDS)
    ids = [item.strip() for item in text.split(",") if item.strip()]
    if not ids:
        raise Pf5Error("--ids is empty")
    return [check_model_id(item) for item in ids]


def product_ids(root: Path = ROOT) -> list[str]:
    path = root / "conf" / "product_static_bs1.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Pf5Error("cannot read %s: %s" % (path, exc)) from exc
    rows = data.get("models")
    if not isinstance(rows, list):
        raise Pf5Error("%s has no models list" % path)
    ids = []
    for row in rows:
        if isinstance(row, dict) and row.get("id"):
            ids.append(check_model_id(str(row["id"])))
    return ids


def run_all_stub(baselines_dir: Path) -> int:
    ids = product_ids()
    present = [model_id for model_id in ids if baseline_path(baselines_dir, model_id).is_file()]
    missing = [model_id for model_id in ids if model_id not in present]
    print(
        "pf5 --ids all is not a full-catalog compare (P1-FC-1). "
        "No RPS numbers are invented for ids without a PF1 baseline. "
        "Use --ids coverage_set for the %d recorded ids. "
        "product ids=%d with_baseline=%d missing=%d"
        % (len(COVERAGE_SET_IDS), len(ids), len(present), len(missing))
    )
    if missing:
        print("missing baselines: %s" % ", ".join(missing))
    return 2


def load_changelog(path: Path) -> dict[str, str]:
    data = load_json(path)
    if data.get("schema") != CHANGELOG_SCHEMA:
        raise Pf5Error("%s schema must be %s" % (path, CHANGELOG_SCHEMA))
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise Pf5Error("%s entries must be a list" % path)
    reasons: dict[str, str] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise Pf5Error("%s entries[%d] must be an object" % (path, index))
        model_id = check_model_id(str(entry.get("id") or ""))
        reason = str(entry.get("reason") or "").strip()
        if not reason:
            raise Pf5Error("%s entries[%d] (%s) has an empty reason" % (path, index, model_id))
        reasons[model_id] = reason
    return reasons


def collect_reasons(ids: list[str], changelog: Path | None, reason_file: str) -> dict[str, str]:
    reasons: dict[str, str] = {}
    if changelog is not None:
        if not changelog.is_file():
            raise Pf5Error("changelog not found: %s" % changelog)
        reasons.update(load_changelog(changelog))
    if reason_file:
        path = Path(reason_file)
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise Pf5Error("cannot read reason file %s: %s" % (path, exc)) from exc
        if not text:
            raise Pf5Error("reason file is empty: %s" % path)
        for model_id in ids:
            reasons[model_id] = text
    return reasons


def import_markdown(markdown: Path, baselines_dir: Path) -> int:
    try:
        text = markdown.read_text(encoding="utf-8")
    except OSError as exc:
        raise Pf5Error("cannot read %s: %s" % (markdown, exc)) from exc
    rows = parse_coverage_markdown(text)
    by_id = {row["id"]: row for row in rows}
    missing = [model_id for model_id in COVERAGE_SET_IDS if model_id not in by_id]
    if missing:
        raise Pf5Error("markdown missing coverage-set ids: %s" % ", ".join(missing))
    wrote = 0
    kept = 0
    for model_id in COVERAGE_SET_IDS:
        doc = baseline_from_markdown_fields(by_id[model_id])
        path = baseline_path(baselines_dir, model_id)
        payload = dump_baseline(doc)
        if path.is_file():
            existing = load_baseline(path)
            comparable = {key: existing.get(key) for key in BASELINE_KEYS}
            if comparable != doc:
                raise Pf5Error(
                    "refusing to overwrite %s: imported values differ from the file "
                    "(baseline not rewritten)" % path
                )
            if path.read_text(encoding="utf-8") != payload:
                atomic_write(path, payload)
                wrote += 1
            else:
                kept += 1
            continue
        atomic_write(path, payload)
        wrote += 1
        print("wrote %s" % path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    print("pf5 import-markdown: wrote=%d unchanged=%d from %s" % (wrote, kept, markdown))
    return 0


def check_baselines(baselines_dir: Path, markdown: Path) -> int:
    try:
        text = markdown.read_text(encoding="utf-8")
    except OSError as exc:
        raise Pf5Error("cannot read %s: %s" % (markdown, exc)) from exc
    parsed = {
        row["id"]: baseline_from_markdown_fields(row) for row in parse_coverage_markdown(text)
    }
    print(
        "pf5 check-baselines: %d coverage-set files under %s"
        % (len(COVERAGE_SET_IDS), baselines_dir)
    )
    for model_id in COVERAGE_SET_IDS:
        path = baseline_path(baselines_dir, model_id)
        if not path.is_file():
            raise Pf5Error("missing baseline %s" % path)
        doc = load_baseline(path)
        if model_id not in parsed:
            raise Pf5Error("markdown has no row for %s" % model_id)
        expected = parsed[model_id]
        if doc.get("source") == "docs/perf/coverage_set_http_rps.md":
            comparable = {key: doc.get(key) for key in BASELINE_KEYS}
            if comparable != expected:
                raise Pf5Error(
                    "%s does not match %s (refusing to treat drifted numbers as the record)"
                    % (path, markdown)
                )
        elif doc.get("source") not in ("coverage_http_report.py", "pf5_retest.py --adopt-baseline"):
            raise Pf5Error("%s has unknown source %r" % (path, doc.get("source")))
        print(
            "  %s success_rps=%s p99_ms=%s worker_nums=%s date=%s hardware=%s"
            % (
                model_id,
                fmt_num(float(doc["success_rps"])),
                fmt_num(float(doc["p99_ms"])),
                doc["worker_nums"],
                doc["date"],
                doc["hardware"],
            )
        )
    changelog = baselines_dir / DEFAULT_CHANGELOG_NAME
    if changelog.is_file():
        load_changelog(changelog)
        print("  changelog %s parses" % changelog.name)
    print("pf5 check-baselines passed")
    return 0


def _load_current_for(model_id: str, current: str, current_dir: str) -> dict:
    if current:
        return load_json(Path(current))
    if current_dir:
        path = Path(current_dir) / ("%s.json" % model_id)
        if not path.is_file():
            raise Pf5Error("missing current metrics %s" % path)
        return load_json(path)
    raise Pf5Error("pass --current, --current-dir, or --dry-compare")


def run_compare(
    ids: list[str],
    baselines_dir: Path,
    current: str,
    current_dir: str,
    hardware_flag: str | None,
    reasons: dict[str, str],
    dry_compare: bool,
) -> int:
    failed = 0
    for model_id in ids:
        path = baseline_path(baselines_dir, model_id)
        if not path.is_file():
            print(
                "FAIL %s no baseline at %s (PF1 report missing; not inventing RPS numbers)"
                % (model_id, path)
            )
            failed += 1
            continue
        before = path.read_bytes()
        try:
            baseline = load_baseline(path)
            if dry_compare:
                current_doc = json.loads(before.decode("utf-8"))
            else:
                current_doc = _load_current_for(model_id, current, current_dir)
            metrics = metrics_from_current(current_doc, model_id, hardware_flag)
            ok, line = compare_metrics(baseline, metrics, reasons.get(model_id))
        except Pf5Error as exc:
            print("FAIL %s %s" % (model_id, exc))
            failed += 1
            if path.read_bytes() != before:
                path.write_bytes(before)
            continue
        print(line)
        if path.read_bytes() != before:
            path.write_bytes(before)
            print("FAIL %s baseline was modified during compare; restored %s" % (model_id, path))
            failed += 1
            continue
        if not ok:
            failed += 1
    print("pf5: %d passed, %d failed" % (len(ids) - failed, failed))
    return 1 if failed else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true", help="run synthetic pass/fail edges and exit")
    parser.add_argument(
        "--check-baselines",
        action="store_true",
        help="parse coverage-set baseline JSON and match the coverage markdown",
    )
    parser.add_argument(
        "--import-markdown",
        action="store_true",
        help="write missing baselines from the coverage markdown; refuse to change recorded numbers",
    )
    parser.add_argument(
        "--adopt-baseline",
        action="store_true",
        help="replace recorded RPS/p99 for --ids from --current, only with a changelog reason and the same hardware",
    )
    parser.add_argument(
        "--dry-compare",
        action="store_true",
        help="compare each selected baseline to itself (no live load)",
    )
    parser.add_argument("--ids", default="", help="coverage_set | all | id[,id...]")
    parser.add_argument("--current", default="", help="metrics JSON for a single --ids value")
    parser.add_argument("--current-dir", default="", help="directory of <id>.json current metrics")
    parser.add_argument("--hardware", default="", help="hardware string when the current JSON omits it")
    parser.add_argument("--baseline-dir", default="", help="default docs/perf/baselines")
    parser.add_argument("--markdown", default="", help="default docs/perf/coverage_set_http_rps.md")
    parser.add_argument(
        "--changelog",
        default="",
        help="regression JSON (schema pf5-regression-v1). Default: baselines/regressions.json when present",
    )
    parser.add_argument(
        "--reason-file",
        default="",
        help="plain-text reason applied to every compared id (intentional RPS/p99 regression only)",
    )
    return parser


def _baselines_dir(args: argparse.Namespace) -> Path:
    return Path(args.baseline_dir) if args.baseline_dir else DEFAULT_BASELINES


def _markdown_path(args: argparse.Namespace) -> Path:
    return Path(args.markdown) if args.markdown else DEFAULT_MARKDOWN


def _explicit_changelog(args: argparse.Namespace, baselines_dir: Path) -> Path | None:
    if args.changelog:
        return Path(args.changelog)
    default = baselines_dir / DEFAULT_CHANGELOG_NAME
    if default.is_file():
        return default
    return None


def _capture_main(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    modes = [
        args.self_test,
        args.check_baselines,
        args.import_markdown,
        args.adopt_baseline,
        bool(args.ids) and not args.adopt_baseline,
    ]
    if sum(bool(item) for item in modes) != 1:
        parser.error("choose one of --self-test, --check-baselines, --import-markdown, --adopt-baseline, --ids")
    if args.adopt_baseline and not args.ids.strip():
        parser.error("--adopt-baseline requires --ids")
    try:
        if args.self_test:
            return _self_test()
        baselines_dir = _baselines_dir(args)
        markdown = _markdown_path(args)
        if args.check_baselines:
            return check_baselines(baselines_dir, markdown)
        if args.import_markdown:
            return import_markdown(markdown, baselines_dir)
        hardware_flag = args.hardware.strip() or None
        if args.adopt_baseline:
            if args.ids.strip() == "all":
                raise Pf5Error("--adopt-baseline does not accept --ids all")
        elif args.ids.strip() == "all":
            return run_all_stub(baselines_dir)
        ids = resolve_ids(args.ids)
        assert ids is not None
        if args.adopt_baseline:
            if args.current and len(ids) != 1:
                raise Pf5Error("--current adopts one id; use --current-dir for several")
            if not args.current and not args.current_dir:
                raise Pf5Error("--adopt-baseline needs --current or --current-dir")
            changelog = Path(args.changelog) if args.changelog else None
            if changelog is None and not args.reason_file:
                default = baselines_dir / DEFAULT_CHANGELOG_NAME
                changelog = default if default.is_file() else None
            reasons = collect_reasons(ids, changelog, args.reason_file)
            for model_id in ids:
                reason = reasons.get(model_id, "")
                if not reason.strip():
                    raise Pf5Error("adopt-baseline: no changelog reason for %s" % model_id)
                current_doc = _load_current_for(model_id, args.current, args.current_dir)
                if "id" not in current_doc and "url" not in current_doc and "load_url" not in current_doc:
                    current_doc = dict(current_doc)
                    current_doc["id"] = model_id
                print(adopt_baseline(current_doc, baselines_dir, reason, hardware_flag))
            return 0
        if not args.dry_compare and not args.current and not args.current_dir:
            raise Pf5Error("pass --current, --current-dir, or --dry-compare")
        if args.current and len(ids) != 1:
            raise Pf5Error("--current compares one id; use --current-dir for several")
        changelog = _explicit_changelog(args, baselines_dir) if (args.changelog or not args.reason_file) else None
        if args.changelog:
            changelog = Path(args.changelog)
        elif args.reason_file:
            changelog = None
        reasons = collect_reasons(ids, changelog, args.reason_file) if (changelog or args.reason_file) else {}
        return run_compare(
            ids,
            baselines_dir,
            args.current,
            args.current_dir,
            hardware_flag,
            reasons,
            args.dry_compare,
        )
    except Pf5Error as exc:
        sys.stderr.write("pf5: %s\n" % exc)
        return 1


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _synthetic_baseline(model_id: str = "yolov8s", **overrides) -> dict:
    load = (
        "python3 scripts/server/http_infer_rps.py --raw --model %s "
        "--gateway http://127.0.0.1:8080 -c 8 -d 60s --warmup 5s" % model_id
    )
    doc = build_baseline(
        model_id=model_id,
        source="pf5 self-test synthetic",
        date="2026-01-01",
        hardware="Synthetic PF5 fixture, 0 MiB",
        worker_nums=4,
        decode_auto="cpu",
        success_rps=1000.0,
        p50_ms=10.0,
        p99_ms=100.0,
        load_command=load,
        load_url="http://127.0.0.1:8080/v1/models/%s/infer" % model_id,
        image=None,
        server_uri=None,
        conf_server=None,
        engine="static shape batch=1",
        wall_s=60.0,
        error_rate=0.0,
    )
    doc.update(overrides)
    return doc


def _self_test() -> int:
    failed = 0

    def check(cond: bool, label: str) -> None:
        nonlocal failed
        if not cond:
            sys.stderr.write("FAIL: %s\n" % label)
            failed += 1

    print("pf5 self-test uses synthetic fixtures, not live GPU measurements")

    quoted = parse_kv_line(
        r'checked=yes | hardware="NVIDIA GeForce RTX 2070 SUPER, 8192 MiB" | '
        r'note="say \"hi\"" | success_rps=1.5 | worker_nums=4'
    )
    check(quoted["hardware"] == RECORDED_HARDWARE, "quoted hardware")
    check(quoted["note"] == 'say "hi"', "escaped quote")
    check(quoted["success_rps"] == "1.5", "unquoted number")

    try:
        markdown_text = DEFAULT_MARKDOWN.read_text(encoding="utf-8")
    except OSError as exc:
        check(False, "read coverage markdown: %s" % exc)
        markdown_text = ""
    if markdown_text:
        rows = parse_coverage_markdown(markdown_text)
        by_id = {row["id"]: row for row in rows}
        check(list(by_id) == list(COVERAGE_SET_IDS), "coverage section order %s" % list(by_id))
        for model_id, token in RECORDED_TOKENS.items():
            fields = by_id[model_id]
            check(fields["success_rps"] == token[0], "%s success_rps token" % model_id)
            check(fields["p99_ms"] == token[1], "%s p99_ms token" % model_id)
            check(fields["worker_nums"] == token[2], "%s worker_nums token" % model_id)
            check(fields["date"] == token[3], "%s date token" % model_id)
            check(fields["decode_auto"] == token[4], "%s decode_auto token" % model_id)
            check(fields["hardware"] == RECORDED_HARDWARE, "%s hardware" % model_id)
            doc = baseline_from_markdown_fields(fields)
            check(doc["success_rps"] == float(token[0]), "%s success_rps float" % model_id)
            check(doc["p99_ms"] == float(token[1]), "%s p99_ms float" % model_id)
            check(doc["concurrency"] == int(token[2]) * 2, "%s concurrency" % model_id)
            check("--raw" in doc["load_command"] and ("--model %s" % model_id) in doc["load_command"],
                  "%s load_command" % model_id)

    base = _synthetic_baseline()
    hw = base["hardware"]

    def current_from(rps: float, p99: float, **overrides) -> dict:
        doc = {
            "id": "yolov8s",
            "success_rps": rps,
            "p99_ms": p99,
            "hardware": hw,
            "worker_nums": 4,
            "concurrency": 8,
            "wall_s": 60.0,
            "warmup_s": 5.0,
        }
        doc.update(overrides)
        return doc

    ok, line = compare_metrics(base, metrics_from_current(current_from(850.0, 150.0), "yolov8s", None), None)
    check(ok and "PASS" in line and "0.85" in line and "1.50" in line, "at floor and ceil: %s" % line)
    ok, line = compare_metrics(base, metrics_from_current(current_from(849.0, 150.0), "yolov8s", None), None)
    check((not ok) and "success_rps" in line and "0.85" in line, "rps below floor: %s" % line)
    ok, line = compare_metrics(base, metrics_from_current(current_from(1000.0, 150.01), "yolov8s", None), None)
    check((not ok) and "p99_ms" in line and "1.50" in line, "p99 above ceil: %s" % line)
    ok, line = compare_metrics(base, metrics_from_current(current_from(849.0, 160.0), "yolov8s", None), None)
    check((not ok) and "success_rps" in line and "p99_ms" in line, "both misses: %s" % line)

    mismatch = current_from(1000.0, 100.0, hardware="Other GPU, 99999 MiB")
    ok, line = compare_metrics(base, metrics_from_current(mismatch, "yolov8s", None), "ignored reason")
    check((not ok) and "hardware mismatch" in line and "not rewritten" in line, "hardware mismatch: %s" % line)

    ok, line = compare_metrics(
        base, metrics_from_current(current_from(849.0, 160.0), "yolov8s", None), "decoder change accepted"
    )
    check(ok and "intentional regression" in line and "decoder change accepted" in line, "changelog override: %s" % line)
    ok, line = compare_metrics(base, metrics_from_current(current_from(849.0, 100.0), "yolov8s", None), "   ")
    check(not ok, "blank reason does not override: %s" % line)

    bad_workers = current_from(1000.0, 100.0, worker_nums=2, concurrency=4)
    ok, line = compare_metrics(
        base, metrics_from_current(bad_workers, "yolov8s", None), "should not excuse command"
    )
    check((not ok) and "worker_nums" in line, "worker_nums mismatch: %s" % line)

    raw_out = {
        "url": "http://127.0.0.1:8080/v1/models/yolov8s/infer",
        "rps": 900.0,
        "latency_ms": {"p50": 11.0, "p99": 110.0},
        "concurrency": 8,
        "wall_s": 60.1,
        "warmup_s": 5.0,
    }
    metrics = metrics_from_current(raw_out, "yolov8s", hw)
    ok, line = compare_metrics(base, metrics, None)
    check(ok, "http_infer_rps --out shape: %s" % line)
    try:
        metrics_from_current(raw_out, "yolov8s", None)
        check(False, "missing hardware must fail")
    except Pf5Error as exc:
        check("no hardware" in str(exc), "missing hardware message: %s" % exc)
    try:
        metrics_from_current(
            {"url": "http://127.0.0.1:8080/v1/models/libface/infer", "rps": 1, "latency_ms": {"p99": 1}, "concurrency": 8},
            "yolov8s",
            hw,
        )
        check(False, "url id mismatch must fail")
    except Pf5Error as exc:
        check("does not match" in str(exc), "url id mismatch: %s" % exc)

    short = current_from(1000.0, 100.0, wall_s=10.0)
    ok, line = compare_metrics(base, metrics_from_current(short, "yolov8s", None), None)
    check((not ok) and "wall_s" in line, "short wall: %s" % line)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base_dir = root / "baselines"
        path = baseline_path(base_dir, "yolov8s")
        atomic_write(path, dump_baseline(base))
        before = path.read_bytes()
        code, out, _err = _capture_main(
            [
                "--ids",
                "yolov8s",
                "--baseline-dir",
                str(base_dir),
                "--current",
                str(_dump_current(root / "low.json", current_from(100.0, 100.0))),
            ]
        )
        check(code == 1 and "FAIL" in out and "success_rps" in out, "cli rps fail exit %s %s" % (code, out))
        check(path.read_bytes() == before, "cli fail rewrote baseline")
        code, out, _err = _capture_main(
            [
                "--ids",
                "yolov8s",
                "--baseline-dir",
                str(base_dir),
                "--current",
                str(_dump_current(root / "hw.json", mismatch)),
            ]
        )
        check(
            code == 1 and "hardware mismatch" in out and "not rewritten" in out,
            "cli hardware exit %s %s" % (code, out),
        )
        check(path.read_bytes() == before, "cli hardware rewrote baseline")
        reason = root / "reason.txt"
        reason.write_text("accepted synthetic regression\n", encoding="utf-8")
        code, out, _err = _capture_main(
            [
                "--ids",
                "yolov8s",
                "--baseline-dir",
                str(base_dir),
                "--current",
                str(_dump_current(root / "low.json", current_from(100.0, 100.0))),
                "--reason-file",
                str(reason),
            ]
        )
        check(code == 0 and "intentional regression" in out, "cli reason exit %s %s" % (code, out))
        check(path.read_bytes() == before, "cli reason rewrote baseline")
        code, out, _err = _capture_main(
            [
                "--ids",
                "yolov8s",
                "--baseline-dir",
                str(base_dir),
                "--current",
                str(_dump_current(root / "hw.json", mismatch)),
                "--reason-file",
                str(reason),
            ]
        )
        check(code == 1 and "hardware mismatch" in out, "cli reason must not excuse hardware (%s) %s" % (code, out))
        check(path.read_bytes() == before, "cli hardware+reason rewrote baseline")

        row = {
            "id": "yolov8s",
            "blank": False,
            "date": "2026-01-02",
            "hardware": hw,
            "worker_nums": 4,
            "decode_auto": "cpu",
            "success_rps": 500.0,
            "p50_ms": 12.0,
            "p99_ms": 120.0,
            "load_command": base["load_command"],
            "load_url": base["load_url"],
            "image": "-",
            "server_uri": None,
            "conf_server": None,
            "engine": "static shape batch=1",
            "wall_s": 60.0,
            "error_rate": 0.0,
        }
        other_dir = root / "record"
        status = record_baseline(row, other_dir)
        check(status == "created", "record created %s" % status)
        created = (other_dir / "yolov8s.json").read_bytes()
        status = record_baseline(dict(row, success_rps=1.0), other_dir)
        check(status == "kept", "record kept %s" % status)
        check((other_dir / "yolov8s.json").read_bytes() == created, "kept rewrote numbers")
        try:
            record_baseline(dict(row, hardware="Different GPU, 1 MiB"), other_dir)
            check(False, "record hardware must raise")
        except BaselineHardwareError as exc:
            check("refusing to rewrite" in str(exc), "record hardware message")
        check((other_dir / "yolov8s.json").read_bytes() == created, "record hardware rewrote file")

        adopt_current = current_from(700.0, 140.0, date="2026-02-02")
        try:
            adopt_baseline(adopt_current, base_dir, "  ", None)
            check(False, "adopt empty reason must fail")
        except Pf5Error:
            pass
        check(path.read_bytes() == before, "empty adopt rewrote baseline")
        try:
            adopt_baseline(dict(adopt_current, hardware="Different GPU, 1 MiB"), base_dir, "because", None)
            check(False, "adopt hardware must fail")
        except BaselineHardwareError:
            pass
        check(path.read_bytes() == before, "adopt hardware rewrote baseline")
        message = adopt_baseline(adopt_current, base_dir, "synthetic adopt", None)
        check("adopted yolov8s" in message, "adopt message %s" % message)
        adopted = load_json(path)
        check(adopted["success_rps"] == 700.0 and adopted["p99_ms"] == 140.0, "adopt wrote numbers")
        check(adopted.get("adopted_reason") == "synthetic adopt", "adopt stored reason")
        check(adopted["hardware"] == hw, "adopt kept hardware")

        code, out, _err = _capture_main(["--ids", "all", "--baseline-dir", str(base_dir)])
        check(code == 2 and "P1-FC-1" in out and "invented" in out, "ids all exit %s %s" % (code, out))

    if DEFAULT_BASELINES.is_dir() and all(
        baseline_path(DEFAULT_BASELINES, model_id).is_file() for model_id in COVERAGE_SET_IDS
    ):
        code, out, err = _capture_main(["--check-baselines"])
        check(code == 0 and "passed" in out, "repo check-baselines exit %s %s %s" % (code, out, err))
        code, out, err = _capture_main(["--ids", "coverage_set", "--dry-compare"])
        check(code == 0 and "0 failed" in out, "repo dry-compare exit %s %s %s" % (code, out, err))
        code, out, _err = _capture_main(["--ids", "all"])
        check(code == 2 and "P1-FC-1" in out, "repo ids all exit %s %s" % (code, out))
        with tempfile.TemporaryDirectory() as drift_tmp:
            drift_root = Path(drift_tmp)
            drifted = DEFAULT_MARKDOWN.read_text(encoding="utf-8").replace(
                "success_rps=894.751", "success_rps=1.0", 1
            )
            report = drift_root / "drift.md"
            report.write_text(drifted, encoding="utf-8")
            copy_dir = drift_root / "baselines"
            copy_dir.mkdir()
            for model_id in COVERAGE_SET_IDS:
                target = baseline_path(copy_dir, model_id)
                target.write_bytes(baseline_path(DEFAULT_BASELINES, model_id).read_bytes())
            watched = baseline_path(copy_dir, "mobilenetv2")
            watched_bytes = watched.read_bytes()
            code, out, err = _capture_main(
                ["--import-markdown", "--markdown", str(report), "--baseline-dir", str(copy_dir)]
            )
            check(
                code == 1 and "refusing to overwrite" in err,
                "import refuses drifted numbers exit %s %s %s" % (code, out, err),
            )
            check(watched.read_bytes() == watched_bytes, "import drift rewrote mobilenetv2")
    else:
        check(False, "checked-in coverage baselines missing; run --import-markdown")

    if failed:
        sys.stderr.write("pf5_retest.py --self-test failed (%d)\n" % failed)
        return 1
    print("pf5_retest.py --self-test passed")
    print(
        "edges: pass-at-threshold, rps-below-floor, p99-above-ceil, "
        "hardware-mismatch, changelog-override, changelog-ignores-hardware, "
        "command-mismatch, http-out-shape, record-keep-refuse, adopt, ids-all-stub"
    )
    return 0


def _dump_current(path: Path, doc: dict) -> Path:
    _write_json(path, doc)
    return path


if __name__ == "__main__":
    sys.exit(main())
