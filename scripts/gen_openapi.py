#!/usr/bin/env python3
"""Generate docs/openapi.json and src/server/openapi_doc.h.

Contract generation chain (single direction, never hand-edited):

  C++ catalogs (param_spec.h + factory/*_task.h)
      -> scripts/contract_dump.cc  -> docs/contract_dump.json   (committed)
      -> this script              -> docs/openapi.json          (committed)
                                   -> src/server/openapi_doc.h  (committed)

The document describes the UNIFIED request envelope (images/params/options,
results[] with per-item status). The legacy img_data field is gone; clients
sending it receive 422 with a migration hint.

Usage:
  python scripts/gen_openapi.py            # regenerate all derived artifacts
  python scripts/gen_openapi.py --check    # fail if outputs are stale or a catalog example is missing or invalid

conf/server [*_SERVER] section names and server_uri feed this script. After
those change, regenerate and commit both output files. port is not an input;
factory server_section / ParamSpec changes also need a new contract_dump.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from repo_toml import load_toml

ROOT = Path(__file__).resolve().parents[1]
DUMP_PATH = ROOT / "docs" / "contract_dump.json"
OPENAPI_PATH = ROOT / "docs" / "openapi.json"
HEADER_PATH = ROOT / "src" / "server" / "openapi_doc.h"


def load_dump() -> dict:
    if not DUMP_PATH.exists():
        sys.exit(
            "ERROR: docs/contract_dump.json missing.\n"
            "Build and run the dump tool first:\n"
            "  cmake --build <full-build-dir> --target contract_dump\n"
            "  <full-build-dir>/bin/contract_dump > docs/contract_dump.json"
        )
    return json.loads(DUMP_PATH.read_text(encoding="utf-8"))


def dump_hash(dump: dict) -> str:
    canonical = json.dumps(dump, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def server_sections() -> list[tuple[str, str, str]]:
    """Return [(server_section, server_uri, category)], one row per server_uri.

    Several catalog ids share one *_SERVER section (yolov8n/s/l/x). Paths in
    the OpenAPI document are those URIs, so collapsing on the section name
    drops every id after the first. CPU and GPU variants of one id share a
    URI; the first file wins.
    """
    found: dict[str, tuple[str, str]] = {}
    conf_server = ROOT / "conf" / "server"
    if not conf_server.exists():
        return []
    for cfg in sorted(conf_server.rglob("*.toml")):
        try:
            table = load_toml(cfg)
        except (ValueError, OSError):
            continue
        for section_name, section in table.items():
            if not isinstance(section, dict):
                continue
            uri = section.get("server_uri")
            if not isinstance(uri, str) or not uri:
                continue
            category = cfg.parent.relative_to(conf_server).parts[0]
            found.setdefault(uri, (section_name, category))
    return sorted((section, uri, category) for uri, (section, category) in found.items())


def dump_entries_by_section(dump: dict) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for task in dump.get("tasks", []):
        for entry in task.get("entries", []):
            entries[entry["server_section"]] = {**entry, "task": task["task"]}
    return entries


def param_schema(params: list[dict]) -> dict:
    properties: dict[str, dict] = {}
    for param in params:
        key = param["key"]
        if param["type"] == "f32":
            prop: dict = {"type": "number", "format": "float"}
        elif param["type"] == "i32":
            prop = {"type": "integer", "format": "int32"}
        elif param["type"] == "bool":
            prop = {"type": "boolean"}
        else:
            prop = {"type": "string"}
        if "description" in param:
            prop["description"] = param["description"]
        if "range" in param:
            prop["minimum"] = param["range"][0]
            prop["maximum"] = param["range"][1]
        if "values" in param:
            prop["enum"] = param["values"]
        if not param.get("request_overridable", True):
            prop["readOnly"] = True
        properties[key] = prop
    return {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
        "description": "Request-level parameter overrides (strict: unknown keys -> 422)",
    }


def request_schema_for(section: str, entry: dict | None) -> dict:
    schema: dict = {
        "type": "object",
        "required": ["images"],
        "additionalProperties": False,
        "properties": {
            "req_id": {"type": "string", "description": "Optional trace id echoed as task_id"},
            "images": {
                "type": "array",
                "items": {"type": "string", "description": "Base64 encoded image"},
                "minItems": 1,
                "description": "One result entry per image (index-aligned results[])",
            },
            "options": {"$ref": "#/components/schemas/OutputOptions"},
        },
    }
    if entry is not None and entry.get("params"):
        schema["properties"]["params"] = {"$ref": "#/components/schemas/Params_%s" % section}
    else:
        schema["properties"]["params"] = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
            "description": "This model declares no request-level parameters",
        }
    return schema


def data_schema_ref(task: str) -> str:
    mapping = {
        "classification": "ClassificationResult",
        "object_detection": "DetectionResult",
        "face_detection": "FaceResult",
        "scene_segmentation": "SegmentationResult",
        "matting": "MattingResult",
        "enhancement": "EnhancementResult",
        "mono_depth_estimation": "DepthResult",
        "feature_point": "FeaturePointResult",
        "ocr": "TextRegionResult",
        "diffusion": "ImageResult",
        "segment_anything_amg": "SamAmgResult",
    }
    return mapping.get(task, "EnvelopeData")


TASK_SCHEMAS: dict[str, dict] = {
    "ClassificationResult": {
        "type": "object",
        "required": ["class_id", "category", "scores"],
        "properties": {
            "class_id": {"type": "integer"},
            "category": {"type": "string"},
            "scores": {"type": "array", "items": {"type": "number"}},
        },
    },
    "BBox": {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 4,
        "maxItems": 4,
        "description": "[x1, y1, x2, y2]",
    },
    "DetectionItem": {
        "type": "object",
        "required": ["class_id", "score", "category", "bbox", "detail_infos"],
        "properties": {
            "class_id": {"type": "integer"},
            "score": {"type": "number"},
            "category": {"type": "string"},
            "bbox": {"$ref": "#/components/schemas/BBox"},
            "detail_infos": {"type": "object"},
        },
    },
    "DetectionResult": {"type": "array", "items": {"$ref": "#/components/schemas/DetectionItem"}},
    "FaceItem": {
        "type": "object",
        "required": ["class_id", "score", "category", "bbox", "landmarks", "detail_infos"],
        "properties": {
            "class_id": {"type": "integer"},
            "score": {"type": "number"},
            "category": {"type": "string"},
            "bbox": {"$ref": "#/components/schemas/BBox"},
            "landmarks": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            },
            "detail_infos": {"type": "object"},
        },
    },
    "FaceResult": {"type": "array", "items": {"$ref": "#/components/schemas/FaceItem"}},
    "TextRegionItem": {
        "type": "object",
        "required": ["score", "bbox", "polygon", "detail_infos"],
        "properties": {
            "score": {"type": "number"},
            "bbox": {"$ref": "#/components/schemas/BBox"},
            "polygon": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            },
            "detail_infos": {"type": "object"},
        },
    },
    "TextRegionResult": {"type": "array", "items": {"$ref": "#/components/schemas/TextRegionItem"}},
    "SegmentationResult": {
        "type": "object",
        "required": ["image", "colorized_mask"],
        "properties": {
            "image": {"type": "string", "description": "Base64 segmentation mask"},
            "colorized_mask": {"type": "string", "description": "Base64 colorized mask"},
        },
    },
    "MattingResult": {
        "type": "object",
        "required": ["image"],
        "properties": {"image": {"type": "string", "description": "Base64 matting result"}},
    },
    "EnhancementResult": {
        "type": "object",
        "required": ["image"],
        "properties": {"image": {"type": "string", "description": "Base64 enhanced image"}},
    },
    "DepthResult": {
        "type": "object",
        "required": ["image"],
        "properties": {"image": {"type": "string", "description": "Base64 colorized depth map"}},
    },
    "FeaturePointItem": {
        "type": "object",
        "required": ["score", "location", "descriptor"],
        "properties": {
            "score": {"type": "number"},
            "location": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            "descriptor": {"type": "array", "items": {"type": "number"}},
        },
    },
    "FeaturePointResult": {"type": "array", "items": {"$ref": "#/components/schemas/FeaturePointItem"}},
    "ImageResult": {
        "type": "object",
        "required": ["image"],
        "properties": {"image": {"type": "string", "description": "Base64 generated image"}},
    },
    "SamAmgItem": {
        "type": "object",
        "required": ["segmentation", "area", "predicted_iou", "stability_score"],
        "properties": {
            "segmentation": {"type": "string", "description": "Base64 mask"},
            "area": {"type": "integer"},
            "bbox": {"$ref": "#/components/schemas/BBox"},
            "predicted_iou": {"type": "number"},
            "stability_score": {"type": "number"},
        },
    },
    "SamAmgResult": {"type": "array", "items": {"$ref": "#/components/schemas/SamAmgItem"}},
}


def model_path_for(section: str, uri: str, entry: dict | None) -> dict:
    task = entry["task"] if entry else "unknown"
    return {
        "post": {
            "summary": entry["display_name"] if entry else "%s inference" % section,
            "tags": [task],
            "security": [{"bearerAuth": []}],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {"schema": {"$ref": "#/components/schemas/Request_%s" % section}},
                    # the same envelope in raw form: the body IS images[0],
                    # params/options ride the X-Mortred-* headers
                    "image/png": {
                        "schema": {"type": "string", "format": "binary"},
                    },
                    "image/jpeg": {
                        "schema": {"type": "string", "format": "binary"},
                    },
                    "application/octet-stream": {
                        "schema": {"type": "string", "format": "binary"},
                    },
                },
            },
            "parameters": [
                {
                    "name": "X-Request-ID",
                    "in": "header",
                    "required": False,
                    "schema": {"type": "string"},
                    "description": "Trace id (raw encoding only; JSON uses req_id)",
                },
                {
                    "name": "X-Mortred-Params",
                    "in": "header",
                    "required": False,
                    "schema": {"type": "string"},
                    "description": 'Compact JSON object, e.g. {"score_threshold":0.35}',
                },
                {
                    "name": "X-Mortred-Options",
                    "in": "header",
                    "required": False,
                    "schema": {"type": "string"},
                    "description": 'Compact JSON object, e.g. {"encoding":"jpeg"}',
                },
            ],
            "responses": {
                "200": {
                    "description": (
                        "Unified envelope; results[] is index-aligned with images[] "
                        "(always length N). Mid-request deadline: HTTP 200 + status 68 + "
                        "partial=true when any item completed; HTTP 504 with N timeout "
                        "slots when none did."
                    ),
                    "content": {
                        "application/json": {
                            "schema": {
                                "allOf": [
                                    {"$ref": "#/components/schemas/UnifiedResponse"},
                                    {
                                        "properties": {
                                            "results": {
                                                "type": "array",
                                                "items": {
                                                    "allOf": [
                                                        {"$ref": "#/components/schemas/ResponseItem"},
                                                        {
                                                            "properties": {
                                                                "data": {
                                                                    "$ref": "#/components/schemas/%s"
                                                                    % data_schema_ref(task)
                                                                }
                                                            }
                                                        },
                                                    ]
                                                },
                                            }
                                        }
                                    },
                                ]
                            }
                        }
                    },
                },
                "400": {"$ref": "#/components/responses/BadRequest"},
                "401": {"$ref": "#/components/responses/Unauthorized"},
                "404": {"$ref": "#/components/responses/NotFound"},
                "405": {"$ref": "#/components/responses/MethodNotAllowed"},
                "413": {"$ref": "#/components/responses/PayloadTooLarge"},
                "415": {"$ref": "#/components/responses/UnsupportedMediaType"},
                "422": {"$ref": "#/components/responses/ValidationError"},
                "429": {"$ref": "#/components/responses/RateLimited"},
                "500": {"$ref": "#/components/responses/InternalError"},
                "504": {"$ref": "#/components/responses/GatewayTimeout"},
            },
        }
    }


def unified_response_schema(options_defaults: dict) -> dict:
    return {
        "type": "object",
        "required": ["status", "status_str", "task_id", "results", "partial"],
        "properties": {
            "status": {"type": "integer", "description": "Business status code (0 = OK)"},
            "status_str": {"type": "string"},
            "task_id": {"type": "string", "description": "req_id echo or server-generated id"},
            "model": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "version": {"type": "string"}},
            },
            "results": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/ResponseItem"},
                "description": (
                    "Index-aligned with the request images[] (always length N for "
                    "admitted inference; unpublished slots are MODEL_RUN_TIMEOUT)"
                ),
            },
            "server_time_ms": {"type": "number"},
            "partial": {"type": "boolean", "description": "true when the deadline hit mid-request"},
            "errors": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/ResponseError"},
                "description": "Present on 422 rejections: pointer-located violations",
            },
        },
    }


def output_options_schema(options_defaults: dict) -> dict:
    return {
        "type": "object",
        "additionalProperties": bool(options_defaults.get("additional_properties", False)),
        "properties": {
            "encoding": {
                "type": "string",
                "enum": ["png", "jpeg", "webp"],
                "default": options_defaults.get("encoding", "png"),
                "description": "Image encoding of embedded outputs",
            },
        },
    }


def error_response(description: str) -> dict:
    return {
        "description": description,
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/UnifiedResponse"}}},
    }


# Path-level examples for the rubric 2026-10-09 HTTP catalog. Ids that share one
# Request_* schema still each get an example on their own server_uri. Paths
# outside this tuple are not required to carry an example.
CATALOG_EXAMPLE_URIS = (
    "/mortred_ai_server_v1/classification/mobilenetv2",
    "/mortred_ai_server_v1/classification/resnet",
    "/mortred_ai_server_v1/classification/densenet",
    "/mortred_ai_server_v1/obj_detection/yolov5l",
    "/mortred_ai_server_v1/obj_detection/yolov6s",
    "/mortred_ai_server_v1/obj_detection/nanodet_1x5",
    "/mortred_ai_server_v1/obj_detection/nanodet_416",
    "/mortred_ai_server_v1/obj_detection/yolov7",
    "/mortred_ai_server_v1/obj_detection/yolov7x",
    "/mortred_ai_server_v1/obj_detection/yolov8n",
    "/mortred_ai_server_v1/obj_detection/yolov8s",
    "/mortred_ai_server_v1/obj_detection/yolov8l",
    "/mortred_ai_server_v1/obj_detection/yolov8x",
    "/mortred_ai_server_v1/obj_detection/libface",
    "/mortred_ai_server_v1/obj_detection/centerface",
    "/mortred_ai_server_v1/scene_segmentation/bisenetv2",
    "/mortred_ai_server_v1/scene_segmentation/pphuman_mobile",
    "/mortred_ai_server_v1/scene_segmentation/pphuman_lite",
    "/mortred_ai_server_v1/scene_segmentation/pphuman_server",
    "/mortred_ai_server_v1/scene_segmentation/hrnet",
    "/mortred_ai_server_v1/scene_segmentation/msocrnet",
    "/mortred_ai_server_v1/ocr/dbnet",
    "/mortred_ai_server_v1/enhancement/attentive_gan",
    "/mortred_ai_server_v1/enhancement/enlightengan",
    "/mortred_ai_server_v1/enhancement/realesrgan",
    "/mortred_ai_server_v1/feature_point/superpoint",
    "/mortred_ai_server_v1/feature_embedding/dinov2_vits14",
    "/mortred_ai_server_v1/feature_embedding/dinov2_vitb14",
    "/mortred_ai_server_v1/feature_embedding/dinov2_vitl14",
    "/mortred_ai_server_v1/matting/modnet",
    "/mortred_ai_server_v1/matting/ppmatting_512",
    "/mortred_ai_server_v1/matting/ppmatting_1024",
    "/mortred_ai_server_v1/matting/ppmatting_resnet34",
    "/mortred_ai_server_v1/matting/ppmatting_v2",
    "/mortred_ai_server_v1/mono_depth_estimation/depth_vits14",
    "/mortred_ai_server_v1/mono_depth_estimation/depth_vitb14",
    "/mortred_ai_server_v1/mono_depth_estimation/depth_vitl14",
    "/mortred_ai_server_v1/mono_depth_estimation/metric3d_512",
    "/mortred_ai_server_v1/mono_depth_estimation/metric3d_1088",
)
_CATALOG_EXAMPLE_URI_SET = frozenset(CATALOG_EXAMPLE_URIS)

_CLASSIFICATION = frozenset({"mobilenetv2", "resnet", "densenet"})
_DETECTION = frozenset(
    {
        "yolov5l",
        "yolov6s",
        "nanodet_1x5",
        "nanodet_416",
        "yolov7",
        "yolov7x",
        "yolov8n",
        "yolov8s",
        "yolov8l",
        "yolov8x",
    }
)
_FACE = frozenset({"libface", "centerface"})
_SEGMENTATION = frozenset(
    {"bisenetv2", "pphuman_mobile", "pphuman_lite", "pphuman_server", "hrnet"}
)
_MATTING = frozenset(
    {"modnet", "ppmatting_512", "ppmatting_1024", "ppmatting_resnet34", "ppmatting_v2"}
)
_ENHANCEMENT = frozenset({"attentive_gan", "enlightengan", "realesrgan"})
_DEPTH = frozenset(
    {"depth_vits14", "depth_vitb14", "depth_vitl14", "metric3d_512", "metric3d_1088"}
)
_DINOV2 = frozenset({"dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"})
_ENVELOPE_DATA = frozenset({"msocrnet"}) | _DINOV2

# Short placeholders. Inner detection/classification/segmentation/OCR/matting/
# enhancement/depth values follow docs/api-contract.md; they are not real images.
_REQ_IMAGE = "base64 encoded image"
_SEG_IMAGE = "<base64-mask>"
_SEG_COLOR = "<base64-colorized-mask>"
_MATTING_IMAGE = "<base64-matting-result>"
_ENHANCE_IMAGE = "<base64-enhanced-image>"
_DEPTH_IMAGE = "<base64-colorized-depth-map>"


def _detection_item() -> dict:
    return {
        "class_id": 0,
        "score": 0.93,
        "category": "person",
        "bbox": [12.0, 40.0, 210.5, 480.0],
        "detail_infos": {},
    }


def _face_item() -> dict:
    item = _detection_item()
    item["landmarks"] = [
        [12.0, 40.0],
        [40.0, 42.0],
        [26.0, 55.0],
        [16.0, 70.0],
        [36.0, 70.0],
    ]
    return item


def _segmentation_data() -> dict:
    return {"image": _SEG_IMAGE, "colorized_mask": _SEG_COLOR}


def example_data(catalog_id: str):
    if catalog_id in _CLASSIFICATION:
        return {"class_id": 281, "category": "tabby cat", "scores": [0.02, 0.91, 0.03]}
    if catalog_id in _DETECTION:
        return [_detection_item()]
    if catalog_id in _FACE:
        return [_face_item()]
    if catalog_id in _SEGMENTATION or catalog_id == "msocrnet":
        return _segmentation_data()
    if catalog_id == "dbnet":
        return [
            {
                "score": 0.97,
                "bbox": [100.0, 200.0, 340.0, 260.0],
                "polygon": [
                    [100.0, 200.0],
                    [340.0, 200.0],
                    [340.0, 260.0],
                    [100.0, 260.0],
                ],
                "detail_infos": {},
            }
        ]
    if catalog_id in _MATTING:
        return {"image": _MATTING_IMAGE}
    if catalog_id in _ENHANCEMENT:
        return {"image": _ENHANCE_IMAGE}
    if catalog_id in _DEPTH:
        return {"image": _DEPTH_IMAGE}
    if catalog_id == "superpoint":
        return [{"score": 0.93, "location": [12.0, 40.0], "descriptor": [0.02, 0.91, 0.03]}]
    if catalog_id in _DINOV2:
        return {"dim": 3, "embedding": [0.02, 0.91, 0.03]}
    raise KeyError(catalog_id)


def request_example(catalog_id: str) -> dict:
    return {"req_id": catalog_id, "images": [_REQ_IMAGE]}


def response_example(catalog_id: str) -> dict:
    return {
        "status": 0,
        "status_str": "OK",
        "task_id": catalog_id,
        "model": {"name": catalog_id, "version": ""},
        "results": [{"status": 0, "data": example_data(catalog_id)}],
        "server_time_ms": 41.2,
        "partial": False,
    }


def _attach_catalog_example(path_item: dict, uri: str) -> None:
    if uri not in _CATALOG_EXAMPLE_URI_SET:
        return
    catalog_id = uri.rstrip("/").rsplit("/", 1)[-1]
    path_item["post"]["requestBody"]["content"]["application/json"]["example"] = request_example(
        catalog_id
    )
    path_item["post"]["responses"]["200"]["content"]["application/json"]["example"] = response_example(
        catalog_id
    )


def _resolve_ref(root: dict, ref: str) -> dict | None:
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    node: object = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, dict) else None


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _type_matches(instance: object, expected: str) -> bool:
    if expected == "object":
        return isinstance(instance, dict)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "boolean":
        return isinstance(instance, bool)
    if expected == "integer":
        return _is_integer(instance)
    if expected == "number":
        return _is_number(instance)
    if expected == "null":
        return instance is None
    return False


def schema_errors(instance: object, schema: dict, root: dict, path: str = "$", ref_stack: tuple[str, ...] = ()) -> list[str]:
    """Validate instance against the OpenAPI keywords this document uses.

    format, description, readOnly, and default do not affect the result.
    """
    if not isinstance(schema, dict):
        return ["%s: schema is not an object" % path]
    ref = schema.get("$ref")
    if isinstance(ref, str):
        if ref in ref_stack:
            return ["%s: cyclic $ref %s" % (path, ref)]
        target = _resolve_ref(root, ref)
        if target is None:
            return ["%s: unresolved $ref %s" % (path, ref)]
        return schema_errors(instance, target, root, path, ref_stack + (ref,))

    errors: list[str] = []
    allof = schema.get("allOf")
    if isinstance(allof, list):
        for sub in allof:
            errors.extend(schema_errors(instance, sub, root, path, ref_stack))

    if instance is None:
        if schema.get("nullable") is True:
            return errors
        if schema.get("nullable") is False or isinstance(schema.get("type"), str) or "enum" in schema:
            enum = schema.get("enum")
            if isinstance(enum, list) and None in enum:
                return errors
            errors.append("%s: null is not allowed" % path)
            return errors
        return errors

    enum = schema.get("enum")
    if isinstance(enum, list) and instance not in enum:
        errors.append("%s: not in enum" % path)

    expected = schema.get("type")
    if isinstance(expected, str) and not _type_matches(instance, expected):
        errors.append("%s: expected type %s" % (path, expected))
        return errors

    if isinstance(instance, dict) and (
        expected == "object" or any(key in schema for key in ("properties", "required", "additionalProperties"))
    ):
        errors.extend(_object_errors(instance, schema, root, path, ref_stack))
    if isinstance(instance, list) and (
        expected == "array" or any(key in schema for key in ("items", "minItems", "maxItems"))
    ):
        errors.extend(_array_errors(instance, schema, root, path, ref_stack))
    if _is_number(instance):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append("%s: below minimum %s" % (path, schema["minimum"]))
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append("%s: above maximum %s" % (path, schema["maximum"]))
    return errors


def _object_errors(instance: dict, schema: dict, root: dict, path: str, ref_stack: tuple[str, ...]) -> list[str]:
    errors: list[str] = []
    props = schema.get("properties", {})
    if not isinstance(props, dict):
        props = {}
    required = schema.get("required", [])
    if isinstance(required, list):
        for key in required:
            if key not in instance:
                errors.append("%s: missing required property '%s'" % (path, key))
    additional = schema.get("additionalProperties", True)
    for key, value in instance.items():
        child = "%s/%s" % (path, key)
        prop_schema = props.get(key)
        if isinstance(prop_schema, dict):
            errors.extend(schema_errors(value, prop_schema, root, child, ref_stack))
        elif additional is False:
            errors.append("%s: additional property '%s'" % (path, key))
        elif isinstance(additional, dict):
            errors.extend(schema_errors(value, additional, root, child, ref_stack))
    return errors


def _array_errors(instance: list, schema: dict, root: dict, path: str, ref_stack: tuple[str, ...]) -> list[str]:
    errors: list[str] = []
    items = schema.get("items")
    if isinstance(items, dict):
        for index, value in enumerate(instance):
            errors.extend(schema_errors(value, items, root, "%s/%d" % (path, index), ref_stack))
    if "minItems" in schema and len(instance) < schema["minItems"]:
        errors.append("%s: fewer than minItems %s" % (path, schema["minItems"]))
    if "maxItems" in schema and len(instance) > schema["maxItems"]:
        errors.append("%s: more than maxItems %s" % (path, schema["maxItems"]))
    return errors


def _response_data_schema_names(schema: dict, root: dict) -> list[str]:
    found: list[str] = []

    def walk(node: object, stack: tuple[str, ...]) -> None:
        if not isinstance(node, dict):
            return
        ref = node.get("$ref")
        if isinstance(ref, str):
            if ref in stack:
                return
            target = _resolve_ref(root, ref)
            if isinstance(target, dict):
                walk(target, stack + (ref,))
            return
        props = node.get("properties")
        if isinstance(props, dict):
            data_schema = props.get("data")
            if isinstance(data_schema, dict) and isinstance(data_schema.get("$ref"), str):
                found.append(data_schema["$ref"].rsplit("/", 1)[-1])
            for sub in props.values():
                walk(sub, stack)
        allof = node.get("allOf")
        if isinstance(allof, list):
            for sub in allof:
                walk(sub, stack)
        items = node.get("items")
        if isinstance(items, dict):
            walk(items, stack)

    walk(schema, ())
    return found


def _is_short_string(value: object) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= 64


def _is_bbox(value: object) -> bool:
    return isinstance(value, list) and len(value) == 4 and all(_is_number(item) for item in value)


def _is_point(value: object) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_is_number(item) for item in value)


def _is_number_array(value: object, low: int, high: int) -> bool:
    return (
        isinstance(value, list)
        and low <= len(value) <= high
        and all(_is_number(item) for item in value)
    )


def _request_field_errors(example: dict, catalog_id: str, label: str) -> list[str]:
    errors: list[str] = []
    if set(example) != {"req_id", "images"}:
        errors.append("%s: request keys must be req_id and images" % label)
    if example.get("req_id") != catalog_id:
        errors.append("%s: req_id must be %s" % (label, catalog_id))
    images = example.get("images")
    if not isinstance(images, list) or len(images) != 1 or not _is_short_string(images[0] if images else None):
        errors.append("%s: images must be one string of length 1..64" % label)
    return errors


def _envelope_field_errors(example: dict, catalog_id: str, label: str) -> list[str]:
    errors: list[str] = []
    if not (_is_integer(example.get("status")) and example.get("status") == 0):
        errors.append("%s: status must be integer 0" % label)
    if example.get("status_str") != "OK":
        errors.append("%s: status_str must be OK" % label)
    if example.get("task_id") != catalog_id:
        errors.append("%s: task_id must be %s" % (label, catalog_id))
    model = example.get("model")
    if not isinstance(model, dict) or model.get("name") != catalog_id:
        errors.append("%s: model.name must be %s" % (label, catalog_id))
    elif not isinstance(model.get("version"), str):
        errors.append("%s: model.version must be a string" % label)
    if not _is_number(example.get("server_time_ms")):
        errors.append("%s: server_time_ms must be a number" % label)
    if example.get("partial") is not False:
        errors.append("%s: partial must be false" % label)
    results = example.get("results")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        errors.append("%s: results must be one object" % label)
        return errors
    item = results[0]
    if not (_is_integer(item.get("status")) and item.get("status") == 0):
        errors.append("%s: results[0].status must be integer 0" % label)
    if item.get("data") is None:
        errors.append("%s: results[0].data must not be null" % label)
    return errors


def _detection_field_errors(item: object, label: str) -> list[str]:
    if not isinstance(item, dict):
        return ["%s: detection item must be an object" % label]
    errors: list[str] = []
    if not _is_integer(item.get("class_id")):
        errors.append("%s: class_id must be an integer" % label)
    if not _is_number(item.get("score")):
        errors.append("%s: score must be a number" % label)
    if not isinstance(item.get("category"), str):
        errors.append("%s: category must be a string" % label)
    if not _is_bbox(item.get("bbox")):
        errors.append("%s: bbox must be four numbers" % label)
    if not isinstance(item.get("detail_infos"), dict):
        errors.append("%s: detail_infos must be an object" % label)
    return errors


def _data_field_errors(catalog_id: str, data: object, label: str) -> list[str]:
    if catalog_id in _CLASSIFICATION:
        if not isinstance(data, dict):
            return ["%s: data must be an object" % label]
        errors: list[str] = []
        if not _is_integer(data.get("class_id")):
            errors.append("%s: class_id must be an integer" % label)
        if not isinstance(data.get("category"), str):
            errors.append("%s: category must be a string" % label)
        if not _is_number_array(data.get("scores"), 1, 8):
            errors.append("%s: scores must be 1..8 numbers" % label)
        return errors
    if catalog_id in _DETECTION or catalog_id in _FACE:
        if not isinstance(data, list) or len(data) != 1:
            return ["%s: data must be one detection item" % label]
        errors = _detection_field_errors(data[0], label)
        if catalog_id in _FACE:
            landmarks = data[0].get("landmarks") if isinstance(data[0], dict) else None
            if (
                not isinstance(landmarks, list)
                or not 1 <= len(landmarks) <= 5
                or not all(_is_point(point) for point in landmarks)
            ):
                errors.append("%s: landmarks must be 1..5 points of two numbers" % label)
        return errors
    if catalog_id in _SEGMENTATION or catalog_id == "msocrnet":
        if not isinstance(data, dict):
            return ["%s: data must be an object" % label]
        errors = []
        if not _is_short_string(data.get("image")) or not _is_short_string(data.get("colorized_mask")):
            errors.append("%s: image and colorized_mask must be strings of length 1..64" % label)
        return errors
    if catalog_id == "dbnet":
        if not isinstance(data, list) or len(data) != 1:
            return ["%s: data must be one text region" % label]
        item = data[0]
        if not isinstance(item, dict):
            return ["%s: text region must be an object" % label]
        errors = []
        if not _is_number(item.get("score")):
            errors.append("%s: score must be a number" % label)
        if not _is_bbox(item.get("bbox")):
            errors.append("%s: bbox must be four numbers" % label)
        polygon = item.get("polygon")
        if (
            not isinstance(polygon, list)
            or not 1 <= len(polygon) <= 4
            or not all(_is_point(point) for point in polygon)
        ):
            errors.append("%s: polygon must be 1..4 points of two numbers" % label)
        if not isinstance(item.get("detail_infos"), dict):
            errors.append("%s: detail_infos must be an object" % label)
        if "text" in item:
            errors.append("%s: text must be absent" % label)
        return errors
    if catalog_id in _MATTING:
        if not isinstance(data, dict):
            return ["%s: data must be an object" % label]
        errors = []
        if not _is_short_string(data.get("image")):
            errors.append("%s: image must be a string of length 1..64" % label)
        if "colorized_mask" in data:
            errors.append("%s: colorized_mask must be absent" % label)
        return errors
    if catalog_id in _ENHANCEMENT or catalog_id in _DEPTH:
        if not isinstance(data, dict) or not _is_short_string(data.get("image")):
            return ["%s: image must be a string of length 1..64" % label]
        return []
    if catalog_id == "superpoint":
        if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
            return ["%s: data must be one feature point" % label]
        item = data[0]
        errors = []
        if not _is_number(item.get("score")):
            errors.append("%s: score must be a number" % label)
        if not _is_point(item.get("location")):
            errors.append("%s: location must be two numbers" % label)
        if not _is_number_array(item.get("descriptor"), 1, 8):
            errors.append("%s: descriptor must be 1..8 numbers" % label)
        return errors
    if catalog_id in _DINOV2:
        if not isinstance(data, dict):
            return ["%s: data must be an object" % label]
        errors = []
        embedding = data.get("embedding")
        dim = data.get("dim")
        if not _is_integer(dim):
            errors.append("%s: dim must be an integer" % label)
        if not _is_number_array(embedding, 1, 8):
            errors.append("%s: embedding must be 1..8 numbers" % label)
        elif _is_integer(dim) and dim != len(embedding):
            errors.append("%s: dim must equal embedding length" % label)
        return errors
    return ["%s: no field contract for %s" % (label, catalog_id)]


def check_catalog_examples(doc: dict) -> list[str]:
    """Existence, schema, and field checks for the 39 catalog path examples."""
    problems: list[str] = []
    if len(CATALOG_EXAMPLE_URIS) != 39 or len(_CATALOG_EXAMPLE_URI_SET) != 39:
        problems.append("catalog example table is not 39 unique paths")
    paths = doc.get("paths", {})
    for uri in CATALOG_EXAMPLE_URIS:
        catalog_id = uri.rstrip("/").rsplit("/", 1)[-1]
        path_item = paths.get(uri)
        if not isinstance(path_item, dict):
            problems.append("%s: path is missing" % uri)
            continue
        post = path_item.get("post", {})
        req_media = post.get("requestBody", {}).get("content", {}).get("application/json", {})
        resp_media = post.get("responses", {}).get("200", {}).get("content", {}).get("application/json", {})
        req_example = req_media.get("example") if isinstance(req_media, dict) else None
        resp_example = resp_media.get("example") if isinstance(resp_media, dict) else None
        if not isinstance(req_media, dict) or "example" not in req_media or not isinstance(req_example, dict):
            problems.append("%s: missing inline request example" % uri)
        else:
            req_schema = req_media.get("schema")
            if not isinstance(req_schema, dict):
                problems.append("%s: request schema is missing" % uri)
            else:
                for err in schema_errors(req_example, req_schema, doc):
                    problems.append("%s request example: %s" % (uri, err))
            problems.extend(_request_field_errors(req_example, catalog_id, uri))
        if not isinstance(resp_media, dict) or "example" not in resp_media or not isinstance(resp_example, dict):
            problems.append("%s: missing inline 200 example" % uri)
            continue
        resp_schema = resp_media.get("schema")
        if not isinstance(resp_schema, dict):
            problems.append("%s: 200 schema is missing" % uri)
        else:
            for err in schema_errors(resp_example, resp_schema, doc):
                problems.append("%s 200 example: %s" % (uri, err))
            if catalog_id in _ENVELOPE_DATA:
                names = _response_data_schema_names(resp_schema, doc)
                if names != ["EnvelopeData"]:
                    problems.append("%s: data $ref is %s; expected EnvelopeData" % (uri, names))
        problems.extend(_envelope_field_errors(resp_example, catalog_id, uri))
        results = resp_example.get("results")
        data = None
        if isinstance(results, list) and results and isinstance(results[0], dict):
            data = results[0].get("data")
        problems.extend(_data_field_errors(catalog_id, data, uri))
    return problems


def build_document() -> dict:
    dump = load_dump()
    entries = dump_entries_by_section(dump)
    options_defaults = dump.get("output_options_defaults", {})

    schemas: dict[str, dict] = {}
    model_paths: dict[str, dict] = {}
    for section, uri, _category in server_sections():
        entry = entries.get(section)
        schemas["Request_%s" % section] = request_schema_for(section, entry)
        if entry is not None and entry.get("params"):
            schemas["Params_%s" % section] = param_schema(entry["params"])
        model_paths[uri] = model_path_for(section, uri, entry)
        _attach_catalog_example(model_paths[uri], uri)

    paths: dict[str, dict] = {
        "/healthz": {
            "get": {"summary": "Liveness probe", "tags": ["ops"], "responses": {"200": {"description": "OK"}}}
        },
        "/ready": {
            "get": {
                "summary": "Readiness probe",
                "tags": ["ops"],
                "responses": {"200": {"description": "Ready"}, "503": {"description": "Not ready"}},
            }
        },
        "/metrics": {
            "get": {
                "summary": "Prometheus metrics",
                "tags": ["ops"],
                "responses": {
                    "200": {
                        "description": "Prometheus text format",
                        "content": {"text/plain": {"schema": {"type": "string"}}},
                    }
                },
            }
        },
        "/openapi.json": {
            "get": {
                "summary": "This OpenAPI document",
                "tags": ["ops"],
                "responses": {
                    "200": {
                        "description": "OpenAPI 3.0 document",
                        "content": {"application/json": {"schema": {"type": "object"}}},
                    }
                },
            }
        },
    }
    paths.update(model_paths)

    schemas["UnifiedResponse"] = unified_response_schema(options_defaults)
    schemas["ResponseItem"] = {
        "type": "object",
        "required": ["status", "data"],
        "properties": {
            "status": {"type": "integer", "description": "Per-item status; 0 = OK"},
            "data": {"nullable": True, "description": "Task payload; null on item failure"},
        },
    }
    schemas["ResponseError"] = {
        "type": "object",
        "required": ["pointer", "message"],
        "properties": {
            "pointer": {"type": "string", "description": "JSON pointer of the offending field"},
            "message": {"type": "string"},
        },
    }
    schemas["OutputOptions"] = output_options_schema(options_defaults)
    schemas["EnvelopeData"] = {"nullable": True}
    schemas.update(TASK_SCHEMAS)

    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Mortred Model Server API",
            "version": "1.0.0",
            "description": (
                "Unified HTTP API for Mortred model servers. Model endpoints require "
                "`Authorization: Bearer <token>` when auth_token is configured. "
                "Request envelope: {req_id, images[], params, options}; response envelope: "
                "{status, status_str, task_id, model, results[], server_time_ms, partial}. "
                "The legacy img_data field was removed: it answers 422 with a migration hint."
            ),
            "x-contract-hash": dump_hash(dump),
        },
        "paths": paths,
        "components": {
            "securitySchemes": {
                "bearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "opaque"}
            },
            "schemas": schemas,
            "responses": {
                "BadRequest": error_response("Malformed JSON body"),
                "Unauthorized": error_response("Missing or invalid bearer token"),
                "NotFound": error_response("Unknown path"),
                "MethodNotAllowed": error_response("Only POST is allowed on model paths"),
                "PayloadTooLarge": error_response(
                    "Body exceeds request_size_limit or images exceeds max_request_items"
                ),
                "UnsupportedMediaType": error_response("Content-Type is not application/json"),
                "ValidationError": error_response(
                    "Strict envelope rejection: errors[] carries JSON pointers"
                ),
                "RateLimited": error_response(
                    "Per-item queue backpressure or per-client-IP rate limit"
                ),
                "InternalError": error_response(
                    "Model or server error (per-item failures keep their own results[].status)"
                ),
                "NotReady": error_response("Server is not ready"),
                "GatewayTimeout": error_response(
                    "Model run timeout: results[] length N, each MODEL_RUN_TIMEOUT"
                ),
            },
        },
    }


def render_header(json_text: str) -> str:
    return (
        "/************************************************\n"
        "* Copyright MaybeShewill-CV. All Rights Reserved.\n"
        "* Author: MaybeShewill-CV\n"
        "* File: openapi_doc.h\n"
        "* Date: 26-8-31\n"
        "************************************************/\n"
        "\n"
        "// GENERATED FILE: do not edit by hand. Regenerate with:\n"
        "//   cmake --build <full-build-dir> --target contract_dump\n"
        "//   <full-build-dir>/bin/contract_dump > docs/contract_dump.json\n"
        "//   python scripts/gen_openapi.py\n"
        "\n"
        "#ifndef MORTRED_SERVER_OPENAPI_DOC_H\n"
        "#define MORTRED_SERVER_OPENAPI_DOC_H\n"
        "\n"
        "#include <string>\n"
        "\n"
        "namespace jinq {\n"
        "namespace server {\n"
        "\n"
        'inline const std::string k_openapi_doc_json = R"MORTRED_OPENAPI(\n'
        + json_text
        + ')MORTRED_OPENAPI";\n'
        "\n"
        "}  // namespace server\n"
        "}  // namespace jinq\n"
        "\n"
        "#endif  // MORTRED_SERVER_OPENAPI_DOC_H\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify files are up to date and catalog examples match their schemas",
    )
    args = parser.parse_args()

    doc = build_document()
    json_text = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
    header_text = render_header(json_text)

    if args.check:
        stale = []
        if not OPENAPI_PATH.exists() or OPENAPI_PATH.read_text(encoding="utf-8") != json_text:
            stale.append("docs/openapi.json is out of date")
        if not HEADER_PATH.exists() or HEADER_PATH.read_text(encoding="utf-8") != header_text:
            stale.append("src/server/openapi_doc.h is out of date")
        example_problems = check_catalog_examples(doc)
        if stale:
            for problem in stale:
                print("ERROR: %s (run: python scripts/gen_openapi.py)" % problem)
            print(
                "After changing a conf/server [*_SERVER] section name or "
                "server_uri, regenerate and commit docs/openapi.json and "
                "src/server/openapi_doc.h. port/worker_nums do not affect "
                "OpenAPI; factory server_section or ParamSpec changes also "
                "need a refreshed docs/contract_dump.json first."
            )
        for problem in example_problems:
            print("ERROR: %s" % problem)
        if stale or example_problems:
            return 1
        print("OpenAPI generation is up to date.")
        return 0

    OPENAPI_PATH.write_text(json_text, encoding="utf-8")
    HEADER_PATH.write_text(header_text, encoding="utf-8")
    print("wrote %s" % OPENAPI_PATH.relative_to(ROOT))
    print("wrote %s" % HEADER_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
