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
  python scripts/gen_openapi.py --check    # fail if anything is out of date
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
    """Return [(server_section, server_uri, category)] from conf/server/*.toml."""
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
            found.setdefault(section_name, (uri, category))
    return sorted((name, uri, category) for name, (uri, category) in found.items())


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
                    "application/json": {"schema": {"$ref": "#/components/schemas/Request_%s" % section}}
                },
            },
            "responses": {
                "200": {
                    "description": (
                        "Unified envelope; results[] aligns with images[]. A mid-request "
                        "deadline returns the completed items with partial=true."
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
                "description": "Index-aligned with the request images[]",
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
        "additionalProperties": False,
        "properties": {
            "encoding": {
                "type": "string",
                "enum": ["png", "jpeg", "webp"],
                "default": options_defaults.get("encoding", "png"),
                "description": "Image encoding of embedded outputs",
            },
            "include_image": {"type": "boolean", "default": options_defaults.get("include_image", True)},
            "max_results": {
                "type": "integer",
                "minimum": 0,
                "default": options_defaults.get("max_results", 0),
                "description": "0 = unlimited",
            },
            "echo_params": {"type": "boolean", "default": options_defaults.get("echo_params", False)},
        },
    }


def error_response(description: str) -> dict:
    return {
        "description": description,
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/UnifiedResponse"}}},
    }


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
        "/welcome": {
            "get": {
                "summary": "Legacy welcome page",
                "deprecated": True,
                "tags": ["legacy"],
                "responses": {"200": {"description": "HTML page"}},
            }
        },
        "/hello_world": {
            "get": {
                "summary": "Legacy health check (kept for web console)",
                "deprecated": True,
                "tags": ["legacy"],
                "responses": {"200": {"description": "HTML page"}},
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
                "GatewayTimeout": error_response("Model run timeout"),
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
    parser.add_argument("--check", action="store_true", help="verify files are up to date")
    args = parser.parse_args()

    doc = build_document()
    json_text = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
    header_text = render_header(json_text)

    if args.check:
        problems = []
        if not OPENAPI_PATH.exists() or OPENAPI_PATH.read_text(encoding="utf-8") != json_text:
            problems.append("docs/openapi.json is out of date")
        if not HEADER_PATH.exists() or HEADER_PATH.read_text(encoding="utf-8") != header_text:
            problems.append("src/server/openapi_doc.h is out of date")
        if problems:
            for p in problems:
                print("ERROR: %s (run: python scripts/gen_openapi.py)" % p)
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