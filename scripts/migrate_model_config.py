#!/usr/bin/env python3
"""Migrate Mortred model TOML files to the unified backend schema.

The parser is the standard-library ``tomllib`` module.  Output is produced by a
small deterministic emitter so this tool has no third-party Python dependency.

Examples::

    python3 scripts/migrate_model_config.py --selftest
    python3 scripts/migrate_model_config.py --dry-run --report report.txt
    python3 scripts/migrate_model_config.py --sections MOBILENETV2 YOLOV8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple



try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    tomllib = None  # type: ignore[assignment]


DEFAULT_CONF_ROOT = Path(__file__).resolve().parent.parent / "conf" / "model"
BACKEND_SUFFIXES = {
    "_TRT": "trt",
    "_ONNX": "onnx",
    "_MNN": "mnn",
}
BACKEND_TYPE_NAMES = {
    "trt": "tensorrt",
    "tensorrt": "tensorrt",
    "onnx": "onnx",
    "mnn": "mnn",
}

# Only these old keys may enter [MODEL.backend].  Every other key stays in
# [MODEL.params] with its original name and value.
BACKEND_SOURCE_KEYS = {
    "model_file_path": "model_file_path",
    "compute_backend": "device",
    "gpu_device_id": "device_id",
    "model_threads_num": "threads",
    "backend_precision_mode": "precision_mode",
    "backend_power_mode": "power_mode",
}

def copy_backend_fields(source_table: Mapping[str, Any], backend: Dict[str, Any]) -> None:
    for old_key, new_key in BACKEND_SOURCE_KEYS.items():
        if old_key not in source_table:
            continue
        value = source_table[old_key]
        if new_key == "device" and isinstance(value, str) and value.lower() == "cuda":
            value = "gpu"
        backend[new_key] = value


BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class MigrationError(RuntimeError):
    """A configuration cannot be safely parsed or migrated."""


@dataclass
class SectionMigration:
    """The result of transforming one top-level model section."""

    section: str
    old_backend: str
    source_section: str
    backend: Dict[str, Any] = field(default_factory=dict)
    extra_backends: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    removed_sections: List[str] = field(default_factory=list)
    key_moves: List[str] = field(default_factory=list)


@dataclass
class FileMigration:
    """All information needed to write and audit one TOML file."""

    path: Path
    original: Dict[str, Any] = field(default_factory=dict)
    migrated: Dict[str, Any] = field(default_factory=dict)
    migrations: List[SectionMigration] = field(default_factory=list)
    untouched_sections: List[str] = field(default_factory=list)
    already_migrated_sections: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def pending(self) -> bool:
        return bool(self.migrations) or has_legacy_markers(self.original)


def require_tomllib() -> None:
    if tomllib is None:
        raise MigrationError(
            "Python 3.11+ is required because this tool uses the standard-library "
            "'tomllib' TOML parser; please run it with python3 >= 3.11"
        )


def load_toml(path: Path) -> Dict[str, Any]:
    require_tomllib()
    try:
        with path.open("rb") as stream:
            value = tomllib.load(stream)
    except FileNotFoundError as exc:
        raise MigrationError(f"file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:  # type: ignore[union-attr]
        raise MigrationError(f"invalid TOML in {path}: {exc}") from exc
    except OSError as exc:
        raise MigrationError(f"cannot read {path}: {exc}") from exc

    if not isinstance(value, dict):
        raise MigrationError(f"expected a TOML table at the document root: {path}")
    return value


def parse_toml_text(text: str) -> Dict[str, Any]:
    require_tomllib()
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:  # type: ignore[union-attr]
        raise MigrationError(f"invalid TOML text: {exc}") from exc


def is_table(value: Any) -> bool:
    return isinstance(value, dict)


def is_migrated_model_section(table: Mapping[str, Any]) -> bool:
    backend = table.get("backend")
    if is_table(backend) and backend.get("type") in {"tensorrt", "onnx", "mnn"}:
        return True
    return any(
        key.endswith("_backend") and is_table(value) and value.get("type") in
        {"tensorrt", "onnx", "mnn"}
        for key, value in table.items()
    )


def is_pure_mnn_section(table: Mapping[str, Any]) -> bool:
    model_path = table.get("model_file_path")
    return (
        isinstance(model_path, str)
        and model_path.lower().endswith((".mnn", ".model"))
        and "compute_backend" in table
    )


def associated_source_sections(data: Mapping[str, Any], section: str) -> List[str]:
    names: List[str] = []
    for suffix in BACKEND_SUFFIXES:
        names.extend(
            name for name in source_section_candidates(section, suffix) if name in data
        )
    return list(dict.fromkeys(names))


def source_section_candidates(section: str, suffix: str) -> List[str]:
    """Return tail-suffix and infix-suffix source names for a model section.

    Most schemas use ``[MODEL_TRT]``.  SAM configs use ``[SAM_TRT_ENCODER]``
    for ``[SAM_ENCODER]``, so also try inserting the backend marker before each
    non-empty role suffix of the model section.
    """

    names = [section + suffix]
    parts = section.split("_")
    marker = suffix.lstrip("_")
    for index in range(1, len(parts)):
        names.append("_".join(parts[:index] + [marker] + parts[index:]))
    return list(dict.fromkeys(names))


def selected_source_section(data: Mapping[str, Any], section: str, suffix: str) -> str:
    return next(
        (name for name in source_section_candidates(section, suffix) if name in data),
        section + suffix,
    )


def has_legacy_markers(data: Mapping[str, Any]) -> bool:
    """Return true when a document still contains obvious old-schema pieces."""

    if "BACKEND_DICT" in data:
        return True
    for value in data.values():
        if not is_table(value):
            continue
        if "backend_type" in value:
            return True
        if is_migrated_model_section(value):
            continue
        if is_pure_mnn_section(value):
            return True
    for section in data:
        if associated_source_sections(data, section):
            return True
    return False


def find_model_candidates(data: Mapping[str, Any]) -> Tuple[List[str], Set[str]]:
    """Return candidate base sections and all source sections they own."""

    candidates: List[str] = []
    for section, value in data.items():
        if not is_table(value):
            continue
        if "backend_type" in value or associated_source_sections(data, section):
            candidates.append(section)
        elif is_pure_mnn_section(value):
            candidates.append(section)

    source_sections: Set[str] = set()
    for candidate in candidates:
        source_sections.update(associated_source_sections(data, candidate))
    return candidates, source_sections


def normalize_backend_type(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return BACKEND_TYPE_NAMES.get(value.strip().lower())


def source_suffix_for_target(target_type: str) -> str:
    old_name = "trt" if target_type == "tensorrt" else target_type
    return next(
        suffix for suffix, backend_name in BACKEND_SUFFIXES.items()
        if backend_name == old_name
    )


def build_key_moves(
    source_table: Mapping[str, Any], base_params: Mapping[str, Any]
) -> List[str]:
    moves: List[str] = []
    for old_key, new_key in BACKEND_SOURCE_KEYS.items():
        if old_key in source_table:
            moves.append(f"{old_key} -> backend.{new_key}")
    for key in base_params:
        if key not in source_table:
            moves.append(f"{key} -> params.{key}")
    for key in source_table:
        if key not in BACKEND_SOURCE_KEYS and key != "input_layout":
            moves.append(f"{key} -> params.{key}")
    return moves


def migrate_lightglue_section(
    data: Mapping[str, Any], section: str, warnings: List[str]
) -> Optional[SectionMigration]:
    """Migrate the two-engine lightglue TRT schema into *_backend tables."""

    source_name = section + "_TRT"
    source_table = data.get(source_name)
    if not is_table(source_table):
        warnings.append(
            f"lightglue section [{section}] has no two-engine [{source_name}] source"
        )
        return None

    backend_common: Dict[str, Any] = {"type": "tensorrt"}
    copy_backend_fields(source_table, backend_common)
    if "compute_backend" not in source_table:
        backend_common["device"] = "cpu"

    engine_paths = {
        "extractor_backend": "extractor_model_file_path",
        "matcher_backend": "matcher_model_file_path",
    }
    extra_backends: Dict[str, Dict[str, Any]] = {}
    for backend_key, path_key in engine_paths.items():
        model_path = source_table.get(path_key)
        if not isinstance(model_path, str) or not model_path:
            warnings.append(
                f"lightglue source [{source_name}] is missing {path_key}; left unchanged"
            )
            return None
        engine_backend = dict(backend_common)
        engine_backend["model_file_path"] = model_path
        extra_backends[backend_key] = engine_backend

    excluded_keys = set(BACKEND_SOURCE_KEYS) | set(engine_paths.values()) | {"input_layout"}
    params: Dict[str, Any] = {
        key: value for key, value in source_table.items() if key not in excluded_keys
    }
    key_moves = [
        f"{path_key} -> {backend_key}.model_file_path"
        for backend_key, path_key in engine_paths.items()
    ]
    key_moves.extend(f"{key} -> params.{key}" for key in params)
    return SectionMigration(
        section=section,
        old_backend="trt",
        source_section=source_name,
        backend={"type": "tensorrt"},
        extra_backends=extra_backends,
        params=params,
        removed_sections=[section] + associated_source_sections(data, section),
        key_moves=key_moves,
    )


def migrate_openai_clip_section(
    data: Mapping[str, Any], warnings: List[str]
) -> Optional[SectionMigration]:
    """Merge the old CLIP encoder sections and tokenizer into [OPENAI_CLIP]."""

    source_names = ["OPENAI_CLIP_VIT_ENCODER", "OPENAI_CLIP_TEXT_ENCODER"]
    source_tables: List[Mapping[str, Any]] = []
    for name in source_names:
        table = data.get(name)
        if not is_table(table):
            warnings.append(f"openai clip source section [{name}] is missing; left unchanged")
            return None
        source_tables.append(table)

    tokenizer_table = data.get("TOKENIZER")
    if not is_table(tokenizer_table) or "vocab_file_path" not in tokenizer_table:
        warnings.append("openai clip [TOKENIZER] is missing or invalid; left unchanged")
        return None

    extra_backends: Dict[str, Dict[str, Any]] = {}
    key_moves: List[str] = []
    for backend_key, source_table, source_name in zip(
        ("visual_backend", "text_backend"), source_tables, source_names
    ):
        model_path = source_table.get("model_file_path")
        if not isinstance(model_path, str) or not model_path:
            warnings.append(
                f"openai clip source [{source_name}] has no model_file_path; left unchanged"
            )
            return None
        backend = {"type": "mnn", "model_file_path": model_path}
        copy_backend_fields(source_table, backend)
        if "compute_backend" not in source_table:
            backend["device"] = "cpu"
        extra_backends[backend_key] = backend
        key_moves.append(f"{source_name}.model_file_path -> {backend_key}.model_file_path")

    params = dict(tokenizer_table)
    for source_table in source_tables:
        for key, value in source_table.items():
            if key not in BACKEND_SOURCE_KEYS and key != "input_layout":
                params[key] = value
    key_moves.extend(f"TOKENIZER.{key} -> params.{key}" for key in params)
    return SectionMigration(
        section="OPENAI_CLIP",
        old_backend="mnn",
        source_section="+".join(source_names),
        backend={"type": "mnn"},
        extra_backends=extra_backends,
        params=params,
        removed_sections=source_names + ["TOKENIZER"],
        key_moves=key_moves,
    )


def infer_backend_type_from_path(model_path: Any) -> Optional[str]:
    if not isinstance(model_path, str):
        return None
    lower_path = model_path.lower()
    if lower_path.endswith(".mnn"):
        return "mnn"
    if lower_path.endswith((".engine", ".plan", ".trt")):
        return "tensorrt"
    if lower_path.endswith(".onnx"):
        return "onnx"
    return None


def sam_backend_from_table(
    source_table: Mapping[str, Any], section: str, warnings: List[str]
) -> Optional[Dict[str, Any]]:
    target_type = normalize_backend_type(source_table.get("backend_type"))
    model_path = source_table.get("model_file_path")
    if target_type is None:
        target_type = infer_backend_type_from_path(model_path)
    if target_type is None or not isinstance(model_path, str) or not model_path:
        warnings.append(
            f"cannot infer backend/model path for sam source [{section}]; left unchanged"
        )
        return None

    backend: Dict[str, Any] = {"type": target_type, "model_file_path": model_path}
    copy_backend_fields(source_table, backend)
    if "compute_backend" not in source_table:
        backend["device"] = "cpu"
    return backend


def sam_source_table(
    data: Mapping[str, Any], base_section: str, warnings: List[str]
) -> Optional[Tuple[str, Mapping[str, Any]]]:
    base_table = data.get(base_section)
    if not is_table(base_table):
        return None
    target_type = normalize_backend_type(base_table.get("backend_type"))
    if target_type is None:
        return base_section, base_table

    suffix = source_suffix_for_target(target_type)
    source_name = selected_source_section(data, base_section, suffix)
    source_table = data.get(source_name)
    if not is_table(source_table):
        warnings.append(
            f"sam section [{base_section}] selects backend '{target_type}' but source "
            f"[{source_name}] is missing; left unchanged"
        )
        return None
    return source_name, source_table


def migrate_sam_sections(
    data: Mapping[str, Any], warnings: List[str]
) -> List[SectionMigration]:
    migrations: List[SectionMigration] = []

    encoder_sources: Tuple[str, ...]
    decoder_sources: Tuple[str, ...]
    if "SAM_VIT_ENCODER" in data or "SAM_VIT_DECODER" in data:
        encoder_sources = ("SAM_VIT_ENCODER",)
        decoder_sources = ("SAM_VIT_DECODER",)
    else:
        encoder_sources = ("SAM_ENCODER",)
        decoder_sources = ("SAM_DECODER",)

    encoder = sam_source_table(data, encoder_sources[0], warnings)
    decoder = sam_source_table(data, decoder_sources[0], warnings)
    if encoder is not None and decoder is not None:
        encoder_source_name, encoder_table = encoder
        decoder_source_name, decoder_table = decoder
        encoder_backend = sam_backend_from_table(
            encoder_table, encoder_source_name, warnings)
        decoder_backend = sam_backend_from_table(
            decoder_table, decoder_source_name, warnings)
        if encoder_backend is not None and decoder_backend is not None:
            removed_sections = [
                encoder_sources[0], decoder_sources[0],
                *associated_source_sections(data, encoder_sources[0]),
                *associated_source_sections(data, decoder_sources[0]),
            ]
            migrations.append(
                SectionMigration(
                    section="SAM_PREDICTOR",
                    old_backend=encoder_backend["type"],
                    source_section=f"{encoder_source_name}+{decoder_source_name}",
                    backend={"type": encoder_backend["type"]},
                    extra_backends={
                        "encoder_backend": encoder_backend,
                        "decoder_backend": decoder_backend,
                    },
                    params={},
                    removed_sections=list(dict.fromkeys(removed_sections)),
                    key_moves=[
                        f"{encoder_source_name}.model_file_path -> encoder_backend.model_file_path",
                        f"{decoder_source_name}.model_file_path -> decoder_backend.model_file_path",
                    ],
                )
            )

    encoder = sam_source_table(data, "SAM_ENCODER", warnings)
    amg_decoder_table = data.get("SAM_AMG_DECODER")
    if encoder is not None and is_table(amg_decoder_table):
        encoder_source_name, encoder_table = encoder
        encoder_backend = sam_backend_from_table(
            encoder_table, encoder_source_name, warnings)
        decoder_backend = sam_backend_from_table(
            amg_decoder_table, "SAM_AMG_DECODER", warnings)
        if encoder_backend is not None and decoder_backend is not None:
            excluded_keys = set(BACKEND_SOURCE_KEYS) | {
                "backend_type", "input_layout", "points_per_size"
            }
            params = {
                key: value for key, value in amg_decoder_table.items()
                if key not in excluded_keys
            }
            if "points_per_size" in amg_decoder_table:
                params["points_per_side"] = amg_decoder_table["points_per_size"]
            removed_sections = [
                "SAM_ENCODER", "SAM_AMG_DECODER",
                *associated_source_sections(data, "SAM_ENCODER"),
            ]
            migrations.append(
                SectionMigration(
                    section="SAM_AMG",
                    old_backend=encoder_backend["type"],
                    source_section=f"{encoder_source_name}+SAM_AMG_DECODER",
                    backend={"type": encoder_backend["type"]},
                    extra_backends={
                        "encoder_backend": encoder_backend,
                        "amg_decoder_backend": decoder_backend,
                    },
                    params=params,
                    removed_sections=list(dict.fromkeys(removed_sections)),
                    key_moves=[
                        f"{encoder_source_name}.model_file_path -> encoder_backend.model_file_path",
                        "SAM_AMG_DECODER.model_file_path -> amg_decoder_backend.model_file_path",
                        *[f"SAM_AMG_DECODER.{key} -> params.{key}" for key in params],
                    ],
                )
            )
    return migrations


def migrate_section(
    data: Mapping[str, Any], section: str, warnings: List[str]
) -> Optional[SectionMigration]:
    base_table = data[section]
    if not is_table(base_table):
        warnings.append(f"section [{section}] is not a table; left unchanged")
        return None
    if is_migrated_model_section(base_table):
        return None

    source_names = associated_source_sections(data, section)
    target_type = normalize_backend_type(base_table.get("backend_type"))
    if target_type is None:
        if is_pure_mnn_section(base_table):
            target_type = "mnn"
        else:
            if "backend_type" in base_table:
                warnings.append(
                    f"section [{section}] has unsupported backend_type="
                    f"{base_table['backend_type']!r}; left unchanged"
                )
            else:
                warnings.append(
                    f"section [{section}] has backend source sections but no "
                    "backend_type; left unchanged"
                )
            return None
    if section == "LIGHTGLUE" and target_type == "tensorrt":
        return migrate_lightglue_section(data, section, warnings)

    old_backend = "trt" if target_type == "tensorrt" else target_type
    source_suffix = source_suffix_for_target(target_type)
    source_name = selected_source_section(data, section, source_suffix)
    source_table: Optional[Mapping[str, Any]] = None
    if source_name in data and is_table(data[source_name]):
        source_table = data[source_name]
    elif source_name not in data and is_pure_mnn_section(base_table):
        source_table = base_table
        source_name = section

    if source_table is None:
        warnings.append(
            f"section [{section}] selects backend '{old_backend}' but source "
            f"section [{source_name}] is missing; left unchanged"
        )
        return None
    if "model_file_path" not in source_table:
        warnings.append(
            f"source section [{source_name}] has no model_file_path; "
            f"section [{section}] left unchanged"
        )
        return None

    backend: Dict[str, Any] = {"type": target_type}
    copy_backend_fields(source_table, backend)
    if "compute_backend" not in source_table:
        backend["device"] = "cpu"

    input_layout = source_table.get("input_layout")
    if isinstance(input_layout, str) and input_layout.strip().lower() == "nchw":
        backend["input_layout"] = "nchw"

    excluded_base_keys = set(BACKEND_SOURCE_KEYS) | {"backend_type", "input_layout"}
    params: Dict[str, Any] = {
        key: value
        for key, value in base_table.items()
        if key not in excluded_base_keys
    }
    for key, value in source_table.items():
        if key == "input_layout":
            if key not in backend:
                params[key] = value
            continue
        if key in BACKEND_SOURCE_KEYS:
            continue
        params[key] = value

    key_moves = build_key_moves(source_table, params)
    if "input_node_size" in params:
        legacy_size = params.pop("input_node_size")
        params.setdefault("model_input_image_size", legacy_size)
        key_moves.append("input_node_size -> params.model_input_image_size")
    return SectionMigration(
        section=section,
        old_backend=old_backend,
        source_section=source_name,
        backend=backend,
        params=params,
        removed_sections=[section] + source_names,
        key_moves=key_moves,
    )


def migrate_document(
    path: Path,
    data: Mapping[str, Any],
    selected_sections: Optional[Set[str]] = None,
) -> FileMigration:
    """Transform one parsed document and return complete audit information."""

    candidates, source_sections = find_model_candidates(data)
    output: Dict[str, Any] = {}
    warnings: List[str] = []
    migrations: List[SectionMigration] = []
    already_migrated: List[str] = []

    sam_migrations = migrate_sam_sections(data, warnings)
    migrations.extend(sam_migrations)
    sam_consumed_sections = {
        name for item in sam_migrations for name in item.removed_sections
    }

    clip_selected = selected_sections is None or "OPENAI_CLIP" in selected_sections
    clip_migration = (
        migrate_openai_clip_section(data, warnings) if clip_selected else None
    )
    if clip_migration is not None:
        migrations.append(clip_migration)
    clip_consumed_sections = (
        set(clip_migration.removed_sections) if clip_migration is not None else set()
    )

    # Surface flat tables that look model-like but do not satisfy the pure-MNN
    # rule from the migration contract.  Their values are copied unchanged.
    for section, value in data.items():
        if (
            section not in candidates
            and section not in source_sections
            and is_table(value)
            and "model_file_path" in value
            and "compute_backend" in value
            and "backend_type" not in value
        ):
            warnings.append(
                f"section [{section}] has model_file_path and compute_backend but "
                "its model path is not .mnn; left unchanged"
            )

    for section in candidates:
        if (
            section in source_sections
            or section in clip_consumed_sections
            or section in sam_consumed_sections
        ):
            continue
        if selected_sections is not None and section.upper() not in selected_sections:
            continue
        migration = migrate_section(data, section, warnings)
        if migration is not None:
            migrations.append(migration)

    migrated_names = {item.section for item in migrations}
    remove_names: Set[str] = {
        name
        for item in migrations
        for name in associated_source_sections(data, item.section)
    }
    for item in migrations:
        remove_names.update(item.removed_sections)
    if migrations:
        remove_names.add("BACKEND_DICT")

    # Keep root scalars first (required by TOML), then stable model-section and
    # other-section order.  This implements "model sections first".
    for section, value in data.items():
        if not is_table(value):
            output[section] = value
    for section in data:
        if section in migrated_names:
            migration = next(item for item in migrations if item.section == section)
            if migration.extra_backends:
                output[section] = {
                    key: dict(value) for key, value in migration.extra_backends.items()
                }
                output[section]["params"] = dict(migration.params)
            else:
                output[section] = {
                    "backend": dict(migration.backend),
                    "params": dict(migration.params),
                }
    for migration in migrations:
        if migration.section in data:
            continue
        if migration.extra_backends:
            output[migration.section] = {
                key: dict(value) for key, value in migration.extra_backends.items()
            }
            output[migration.section]["params"] = dict(migration.params)
        else:
            output[migration.section] = {
                "backend": dict(migration.backend),
                "params": dict(migration.params),
            }
    for section, value in data.items():
        if not is_table(value) or section in migrated_names or section in remove_names:
            continue
        output[section] = value
        if is_migrated_model_section(value):
            already_migrated.append(section)

    untouched = [
        section
        for section in output
        if section not in migrated_names
        and (not is_table(output[section]) or not is_migrated_model_section(output[section]))
    ]
    return FileMigration(
        path=path,
        original=dict(data),
        migrated=output,
        migrations=migrations,
        untouched_sections=untouched,
        already_migrated_sections=already_migrated,
        warnings=warnings,
    )


def quote_key_if_needed(key: str) -> str:
    if BARE_KEY_RE.fullmatch(key):
        return key
    return json.dumps(key, ensure_ascii=False)


def render_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return repr(value)


def render_inline_table(value: Mapping[str, Any]) -> str:
    parts = [
        f"{quote_key_if_needed(str(key))} = {render_inline_value(item)}"
        for key, item in value.items()
    ]
    return "{" + ", ".join(parts) + "}"


def render_inline_value(value: Any) -> str:
    if value is None:
        raise MigrationError("TOML cannot represent None")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return render_float(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        return value.isoformat()
    if isinstance(value, list):
        return "[" + ", ".join(render_inline_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return render_inline_table(value)
    raise MigrationError(f"cannot emit TOML value of type {type(value).__name__}")


def render_multiline_array(lines: List[str], indent: str, values: Sequence[Any]) -> None:
    nested_indent = indent + "  "
    for value in values:
        lines.append(nested_indent + render_inline_value(value) + ",")
    lines.append(indent + "]")


def append_value(lines: List[str], indent: str, key: str, value: Any) -> None:
    prefix = indent + quote_key_if_needed(key) + " = "
    if isinstance(value, list) and value:
        inline = render_inline_value(value)
        if len(value) >= 8 or len(prefix) + len(inline) > 96:
            lines.append(prefix + "[")
            render_multiline_array(lines, indent + "  ", value)
            return
    lines.append(prefix + render_inline_value(value))


def append_table(lines: List[str], path: Tuple[str, ...], table: Mapping[str, Any]) -> None:
    header = ".".join(quote_key_if_needed(part) for part in path)
    lines.append("")
    lines.append(f"[{header}]")

    scalar_keys = [key for key, value in table.items() if not is_table(value)]
    nested_keys = [key for key, value in table.items() if is_table(value)]
    for key in scalar_keys:
        append_value(lines, "", key, table[key])
    for key in nested_keys:
        append_table(lines, path + (key,), table[key])


def emit_toml(data: Mapping[str, Any]) -> str:
    """Emit a deterministic, valid TOML document without external libraries."""

    lines: List[str] = []
    root_scalars = [key for key, value in data.items() if not is_table(value)]
    tables = [(key, value) for key, value in data.items() if is_table(value)]
    for key in root_scalars:
        append_value(lines, "", key, data[key])
    for key, table in tables:
        append_table(lines, (key,), table)
    if not lines:
        lines.append("")
    return "\n".join(lines) + "\n"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old_mode = path.stat().st_mode & 0o777 if path.exists() else None
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if old_mode is not None:
            os.chmod(temporary_path, old_mode)
        os.replace(temporary_path, path)
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def discover_toml_files(root: Path) -> List[Path]:
    if not root.is_dir():
        raise MigrationError(f"configuration root is not a directory: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".toml"
    )


def backend_counter(results: Sequence[FileMigration]) -> Counter:
    counter: Counter = Counter()
    for result in results:
        for migration in result.migrations:
            counter[migration.backend["type"]] += 1
    return counter


def describe_migration(result: FileMigration) -> List[str]:
    lines: List[str] = []
    for migration in result.migrations:
        source_note = (
            f" (source [{migration.source_section}])"
            if migration.source_section != migration.section
            else " (flat source)"
        )
        backend_keys = (
            list(migration.extra_backends)
            if migration.extra_backends
            else ["backend"]
        )
        lines.append(
            f"  [{migration.section}] {migration.old_backend}{source_note} -> "
            + ", ".join(f"[{migration.section}.{key}]" for key in backend_keys)
            + f" type={migration.backend['type']}, "
            f"[{migration.section}.params] keys={len(migration.params)}"
        )
    return lines


def report_text(results: Sequence[FileMigration], root: Path) -> str:
    backends = backend_counter(results)
    warning_count = sum(len(result.warnings) for result in results)
    multi_model_files = sum(len(result.migrations) > 1 for result in results)
    pending_files = sum(result.pending for result in results)

    lines = [
        "Mortred model configuration migration report",
        f"Root: {root}",
        f"Files: {len(results)}",
        "Model sections: "
        + ", ".join(
            f"{name}={backends[name]}" for name in ("tensorrt", "onnx", "mnn")
        ),
        f"Files with multiple migrated model sections: {multi_model_files}",
        f"Warnings: {warning_count}",
        f"Files pending migration: {pending_files}",
        "",
    ]
    for result in results:
        relative = result.path.relative_to(root)
        lines.append(f"FILE {relative}")
        if result.migrations:
            lines.append("  Status: pending migration")
        elif result.pending:
            lines.append(
                "  Status: pending migration (legacy marker without a safe section migration)"
            )
        else:
            lines.append("  Status: already migrated / no old-schema model section")

        for migration in result.migrations:
            backend_keys = (
                list(migration.extra_backends)
                if migration.extra_backends
                else ["backend"]
            )
            lines.extend(
                [
                    f"  SECTION [{migration.section}]",
                    "    Old: backend_type={}; source=[{}]".format(
                        migration.old_backend, migration.source_section
                    ),
                    "    New: "
                    + ", ".join(f"[{migration.section}.{key}]" for key in backend_keys)
                    + f", [{migration.section}.params]",
                    f"    Backend type: {migration.backend['type']}",
                ]
            )
            for move in migration.key_moves:
                lines.append(f"    Key: {move}")
            source_tables = associated_source_sections(result.original, migration.section)
            lines.append(
                "    Remove: "
                + (", ".join(f"[{name}]" for name in source_tables) or "none")
            )
            lines.append(f"    Remove key: [{migration.section}].backend_type")
            lines.append("    Remove table: [BACKEND_DICT]")

        if result.already_migrated_sections:
            lines.append(
                "  Already migrated: "
                + ", ".join(f"[{name}]" for name in result.already_migrated_sections)
            )
        if result.untouched_sections:
            lines.append(
                "  Untouched: "
                + ", ".join(f"[{name}]" for name in result.untouched_sections)
            )
        for warning in result.warnings:
            lines.append(f"  WARNING: {warning}")
        lines.append("")
    return "\n".join(lines)


def print_summary(results: Sequence[FileMigration], root: Path, dry_run: bool) -> None:
    backends = backend_counter(results)
    warning_count = sum(len(result.warnings) for result in results)
    multi_model_files = sum(len(result.migrations) > 1 for result in results)
    pending_files = sum(
        (result.pending if dry_run else has_legacy_markers(result.migrated))
        for result in results
    )
    action = "would migrate" if dry_run else "migrated"
    print(
        f"{action} {sum(len(item.migrations) for item in results)} section(s) in "
        f"{sum(bool(item.migrations) for item in results)} file(s) under {root}"
    )
    print(
        "backends: "
        + ", ".join(
            f"{name}={backends[name]}" for name in ("tensorrt", "onnx", "mnn")
        )
    )
    print(f"multi-model files: {multi_model_files}")
    print(f"warnings: {warning_count}")
    print(f"files pending migration: {pending_files}/{len(results)}")


def migrate_tree(
    root: Path,
    selected_sections: Optional[Sequence[str]] = None,
    dry_run: bool = False,
    write_report: Optional[Path] = None,
) -> Tuple[List[FileMigration], int]:
    require_tomllib()
    selected = (
        {section.strip().upper() for section in selected_sections if section.strip()}
        if selected_sections is not None
        else None
    )
    paths = discover_toml_files(root)
    if not paths:
        raise MigrationError(f"no TOML files found under {root}")

    results: List[FileMigration] = []
    failures: List[str] = []
    for path in paths:
        try:
            data = load_toml(path)
            result = migrate_document(path, data, selected)
            results.append(result)
            if result.migrations and not dry_run:
                atomic_write_text(path, emit_toml(result.migrated))
        except MigrationError as exc:
            failures.append(str(exc))

    if failures:
        raise MigrationError("\n".join(failures))
    if write_report is not None:
        atomic_write_text(write_report, report_text(results, root))
    return results, 0


def run_check(root: Path, write_report: Optional[Path] = None) -> int:
    results, _ = migrate_tree(root, dry_run=True, write_report=write_report)
    for result in results:
        if not result.pending:
            continue
        detail = ", ".join(
            f"[{item.section}] {item.old_backend}->{item.backend['type']}"
            for item in result.migrations
        ) or "legacy marker"
        print(f"OLD {result.path}: {detail}")
    pending = sum(result.pending for result in results)
    migrated = len(results) - pending
    print(f"{pending} file(s) still use the old schema; {migrated} already migrated")
    return 1 if pending else 0


def run_dry_run(root: Path, write_report: Optional[Path] = None) -> int:
    results, _ = migrate_tree(root, dry_run=True, write_report=write_report)
    for result in results:
        file_lines = describe_migration(result)
        if file_lines:
            print(f"DRY RUN {result.path}:")
            print("\n".join(file_lines))
        for warning in result.warnings:
            print(f"WARNING {result.path}: {warning}")
    print_summary(results, root, dry_run=True)
    return 0


def run_migration(root: Path, write_report: Optional[Path] = None) -> int:
    results, _ = migrate_tree(root, dry_run=False, write_report=write_report)
    for result in results:
        for migration in result.migrations:
            print(
                f"MIGRATED {result.path}: [{migration.section}] -> "
                f"[{migration.section}.backend]/[{migration.section}.params]"
            )
        for warning in result.warnings:
            print(f"WARNING {result.path}: {warning}")
    print_summary(results, root, dry_run=False)
    return 0


SELFTEST_TOML = r"""
[DDPM_SAMPLER]
beta_schedule="linear"
timesteps=1000
beta_start=0.0001
beta_end=0.02

[YOLOV8]
backend_type="trt"

[YOLOV8_TRT]
model_file_path="../weights/yolov8s.engine"
model_score_threshold=0.25
input_node_size=[640, 640]
class_names=["one", "two", "three", "four", "five", "six", "seven", "eight"]
profile=[[1, 512, 2], [1, 2048, 2]]
use_explicit_nchw=false

[YOLOV8_ONNX]
model_file_path="../weights/yolov8s.onnx"
compute_backend="cuda"

[LIGHTGLUE]
backend_type="trt"

[LIGHTGLUE_ONNX]
model_file_path="../weights/lightglue.onnx"
compute_backend="cuda"

[LIGHTGLUE_TRT]
extractor_model_file_path="../weights/extractor.engine"
matcher_model_file_path="../weights/matcher.engine"
compute_backend="cuda"
gpu_device_id=0
extract_score_thresh=0.1
match_score_thresh=0.5
long_side_length=512.0

[DDPM_UNET]
backend_type="onnx"

[DDPM_UNET_ONNX]
model_file_path="../weights/ddpm.onnx"
compute_backend="cuda"
gpu_device_id=2
model_threads_num=4
timesteps=1000

[SAM_ENCODER]
backend_type="trt"

[SAM_TRT_ENCODER]
model_file_path="../weights/sam_encoder.engine"

[SAM_ONNX_ENCODER]
model_file_path="../weights/sam_encoder.onnx"

[SAM_DECODER]
backend_type="trt"

[SAM_TRT_DECODER]
model_file_path="../weights/sam_decoder.engine"

[SAM_AMG_DECODER]
model_file_path="../weights/sam_amg_decoder.engine"
worker_queue_size=8
compute_threads=-1
points_per_size=32
pred_iou_thresh=0.88
stability_score_thresh=0.95
box_nms_thresh=0.7
min_mask_region_area=0

[MNET]
backend_type="mnn"

[MNET_MNN]
model_file_path="../weights/model.mnn"
compute_backend="cpu"
model_threads_num=2
backend_precision_mode=1
backend_power_mode=2
input_layout="nchw"
model_input_image_size=[224, 224]

[MOBILENETV2]
model_file_path="../weights/mobilenetv2.mnn"
compute_backend="cuda"
model_threads_num=4
backend_precision_mode=0
backend_power_mode=0
model_score_threshold=0.1
class_name_file="../conf/classes.txt"

[LIBFACE]
model_file_path="../weights/libface.model"
compute_backend="cuda"
model_threads_num=1
model_score_threshold=0.75
model_nms_threshold=0.35
model_keep_top_k=250

[OPENAI_CLIP_VIT_ENCODER]
model_file_path="../weights/visual.mnn"
compute_backend="cuda"
model_threads_num=4
backend_precision_mode=0
backend_power_mode=0

[OPENAI_CLIP_TEXT_ENCODER]
model_file_path="../weights/textual.mnn"
compute_backend="cuda"
model_threads_num=4
backend_precision_mode=0
backend_power_mode=0

[TOKENIZER]
vocab_file_path="../weights/bpe_simple_vocab_16e6.txt"
context_length=77
truncate_context=true

[LEGACY_MODEL]
model_file_path="../weights/legacy.bin"
compute_backend="cuda"
keep_me=true

[BACKEND_DICT]
trt=0
onnx=1
mnn=2
"""


def assert_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def run_selftest() -> int:
    require_tomllib()
    original = parse_toml_text(SELFTEST_TOML)
    result = migrate_document(Path("selftest.toml"), original)

    assert_equal(len(result.migrations), 9, "nine model sections should migrate")
    assert_equal(
        sorted(migration.section for migration in result.migrations),
        [
            "DDPM_UNET",
            "LIBFACE",
            "LIGHTGLUE",
            "MNET",
            "MOBILENETV2",
            "OPENAI_CLIP",
            "SAM_AMG",
            "SAM_PREDICTOR",
            "YOLOV8",
        ],
        "migrated section set",
    )
    assert_equal(
        result.migrated["DDPM_SAMPLER"], original["DDPM_SAMPLER"], "sampler section"
    )
    assert_equal(
        result.migrated["LEGACY_MODEL"], original["LEGACY_MODEL"], "legacy section"
    )
    assert "BACKEND_DICT" not in result.migrated
    assert_equal(
        set(result.migrated["YOLOV8"].keys()), {"backend", "params"}, "empty model table"
    )

    expected_yolo_params = {
        key: value
        for key, value in original["YOLOV8_TRT"].items()
        if key != "model_file_path"
    }
    expected_yolo_params["model_input_image_size"] = expected_yolo_params.pop(
        "input_node_size"
    )
    assert_equal(
        result.migrated["YOLOV8"]["backend"],
        {
            "type": "tensorrt",
            "model_file_path": "../weights/yolov8s.engine",
            "device": "cpu",
        },
        "TRT backend",
    )
    assert_equal(
        set(result.migrated["LIGHTGLUE"].keys()),
        {"extractor_backend", "matcher_backend", "params"},
        "lightglue multi-engine section keys",
    )
    assert_equal(
        result.migrated["LIGHTGLUE"]["extractor_backend"],
        {
            "type": "tensorrt",
            "model_file_path": "../weights/extractor.engine",
            "device": "gpu",
            "device_id": 0,
        },
        "lightglue extractor backend",
    )
    assert_equal(
        result.migrated["LIGHTGLUE"]["matcher_backend"]["model_file_path"],
        "../weights/matcher.engine",
        "lightglue matcher path",
    )
    assert_equal(
        result.migrated["LIGHTGLUE"]["params"],
        {
            "extract_score_thresh": 0.1,
            "match_score_thresh": 0.5,
            "long_side_length": 512.0,
        },
        "lightglue params",
    )
    assert_equal(result.migrated["YOLOV8"]["params"], expected_yolo_params, "TRT params")

    assert_equal(
        result.migrated["DDPM_UNET"]["backend"],
        {
            "type": "onnx",
            "model_file_path": "../weights/ddpm.onnx",
            "device": "gpu",
            "device_id": 2,
            "threads": 4,
        },
        "ONNX backend",
    )
    assert_equal(
        result.migrated["DDPM_UNET"]["params"], {"timesteps": 1000}, "ONNX params"
    )
    assert_equal(
        set(result.migrated["SAM_PREDICTOR"].keys()),
        {"encoder_backend", "decoder_backend", "params"},
        "SAM predictor multi-engine section keys",
    )
    assert_equal(
        result.migrated["SAM_PREDICTOR"]["encoder_backend"],
        {
            "type": "tensorrt",
            "model_file_path": "../weights/sam_encoder.engine",
            "device": "cpu",
        },
        "SAM infix-source encoder backend",
    )
    assert_equal(
        result.migrated["SAM_PREDICTOR"]["decoder_backend"]["model_file_path"],
        "../weights/sam_decoder.engine",
        "SAM infix-source decoder path",
    )
    assert_equal(
        result.migrated["SAM_AMG"]["amg_decoder_backend"]["model_file_path"],
        "../weights/sam_amg_decoder.engine",
        "SAM AMG decoder path",
    )
    assert_equal(
        result.migrated["SAM_AMG"]["params"],
        {
            "worker_queue_size": 8,
            "compute_threads": -1,
            "points_per_side": 32,
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.95,
            "box_nms_thresh": 0.7,
            "min_mask_region_area": 0,
        },
        "SAM AMG params",
    )
    vit_original = parse_toml_text(
        """
        [SAM_VIT_ENCODER]
        model_file_path="../weights/sam_vit_encoder.mnn"
        compute_backend="cuda"
        model_threads_num=4

        [SAM_VIT_DECODER]
        model_file_path="../weights/sam_vit_decoder.onnx"
        compute_backend="cpu"
        model_threads_num=4
        """
    )
    vit_result = migrate_document(Path("sam-vit-selftest.toml"), vit_original)
    assert_equal(len(vit_result.migrations), 1, "one SAM ViT model should migrate")
    assert_equal(
        vit_result.migrated["SAM_PREDICTOR"]["encoder_backend"],
        {
            "type": "mnn",
            "model_file_path": "../weights/sam_vit_encoder.mnn",
            "device": "gpu",
            "threads": 4,
        },
        "SAM ViT encoder backend",
    )
    assert_equal(
        vit_result.migrated["SAM_PREDICTOR"]["decoder_backend"],
        {
            "type": "onnx",
            "model_file_path": "../weights/sam_vit_decoder.onnx",
            "device": "cpu",
            "threads": 4,
        },
        "SAM ViT decoder backend",
    )
    assert_equal(
        result.migrated["MNET"]["backend"],
        {
            "type": "mnn",
            "model_file_path": "../weights/model.mnn",
            "device": "cpu",
            "threads": 2,
            "precision_mode": 1,
            "power_mode": 2,
            "input_layout": "nchw",
        },
        "MNN backend",
    )
    assert_equal(
        result.migrated["LIBFACE"]["backend"],
        {
            "type": "mnn",
            "model_file_path": "../weights/libface.model",
            "device": "gpu",
            "threads": 1,
        },
        "pure MNN .model backend",
    )
    assert_equal(
        result.migrated["LIBFACE"]["params"],
        {
            "model_score_threshold": 0.75,
            "model_nms_threshold": 0.35,
            "model_keep_top_k": 250,
        },
        "pure MNN .model params",
    )
    assert_equal(
        set(result.migrated["OPENAI_CLIP"].keys()),
        {"visual_backend", "text_backend", "params"},
        "openai clip multi-engine section keys",
    )
    assert_equal(
        result.migrated["OPENAI_CLIP"]["visual_backend"],
        {
            "type": "mnn",
            "model_file_path": "../weights/visual.mnn",
            "device": "gpu",
            "threads": 4,
            "precision_mode": 0,
            "power_mode": 0,
        },
        "openai clip visual backend",
    )
    assert_equal(
        result.migrated["OPENAI_CLIP"]["text_backend"]["model_file_path"],
        "../weights/textual.mnn",
        "openai clip text model path",
    )
    assert_equal(
        result.migrated["OPENAI_CLIP"]["params"],
        {
            "vocab_file_path": "../weights/bpe_simple_vocab_16e6.txt",
            "context_length": 77,
            "truncate_context": True,
        },
        "openai clip tokenizer params",
    )
    assert "TOKENIZER" not in result.migrated
    assert "OPENAI_CLIP_VIT_ENCODER" not in result.migrated
    assert "OPENAI_CLIP_TEXT_ENCODER" not in result.migrated
    assert_equal(
        result.migrated["MNET"]["params"],
        {"model_input_image_size": [224, 224]},
        "MNN params",
    )
    assert_equal(
        result.migrated["MOBILENETV2"]["backend"],
        {
            "type": "mnn",
            "model_file_path": "../weights/mobilenetv2.mnn",
            "device": "gpu",
            "threads": 4,
            "precision_mode": 0,
            "power_mode": 0,
        },
        "pure-MNN backend",
    )
    assert_equal(
        result.migrated["MOBILENETV2"]["params"],
        {
            "model_score_threshold": 0.1,
            "class_name_file": "../conf/classes.txt",
        },
        "pure-MNN params",
    )
    assert any("LEGACY_MODEL" in warning for warning in result.warnings)

    emitted = emit_toml(result.migrated)
    reparsed = parse_toml_text(emitted)
    assert_equal(reparsed, result.migrated, "emitter round-trip")

    idempotent = migrate_document(Path("selftest-round2.toml"), reparsed)
    assert_equal(idempotent.migrations, [], "second migration should be a no-op")
    assert_equal(emit_toml(idempotent.migrated), emitted, "idempotent output")
    assert_equal(idempotent.pending, False, "second check should be clean")

    print("OK")
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--conf-root", type=Path, default=DEFAULT_CONF_ROOT)
    parser.add_argument("--check", action="store_true", help="report old-schema files only")
    parser.add_argument("--dry-run", action="store_true", help="do not write TOML files")
    parser.add_argument(
        "--sections",
        nargs="+",
        metavar="SECTION",
        help="only migrate these top-level model sections",
    )
    parser.add_argument(
        "--report", type=Path, help="write a human-readable mapping report"
    )
    parser.add_argument("--selftest", action="store_true", help="run embedded semantic tests")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        if args.selftest:
            return run_selftest()
        root = args.conf_root.expanduser().resolve()
        if args.check:
            return run_check(root, args.report)
        if args.dry_run:
            return run_dry_run(root, args.report)
        return run_migration(root, args.report)
    except MigrationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
