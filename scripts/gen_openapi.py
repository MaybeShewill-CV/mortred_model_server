#!/usr/bin/env python3
"""Generate docs/openapi.json and src/server/openapi_doc.h from conf/server.

The OpenAPI document is the single source of truth for the HTTP API contract:
- every conf/server/*.toml `server_uri` becomes a POST path;
- model paths require Bearer auth (components.securitySchemes.bearerAuth);
- per-task `data` schemas mirror src/server/response_serializers.h;
- the embedded header served at GET /openapi.json stays byte-identical to the
  JSON document (enforced by scripts/check_consistency.py).

Usage:
  python scripts/gen_openapi.py            # regenerate both files
  python scripts/gen_openapi.py --check    # fail if files are out of date
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_toml import load_toml

ROOT = Path(__file__).resolve().parents[1]
OPENAPI_PATH = ROOT / "docs" / "openapi.json"
HEADER_PATH = ROOT / "src" / "server" / "openapi_doc.h"


def server_uris() -> list[tuple[str, str]]:
    """Return [(server_uri, category)] from conf/server/*.toml, deduplicated."""
    found: dict[str, str] = {}
    conf_server = ROOT / "conf" / "server"
    if not conf_server.exists():
        return []
    for cfg in sorted(conf_server.rglob("*.toml")):
        try:
            table = load_toml(cfg)
        except (ValueError, OSError):
            continue
        for section in table.values():
            if not isinstance(section, dict):
                continue
            uri = section.get("server_uri")
            if not isinstance(uri, str) or not uri:
                continue
            category = cfg.parent.relative_to(conf_server).parts[0]
            found.setdefault(uri, category)
    return sorted(found.items())


def data_schema_ref(category: str) -> str:
    mapping = {
        "classification": "ClassificationResult",
        "object_detection": "DetectionResult",
        "scene_segmentation": "SegmentationResult",
        "matting": "MattingResult",
        "enhancement": "EnhancementResult",
        "mono_depth_estimation": "DepthResult",
        "feature_point": "FeaturePointResult",
        "ocr": "TextRegionResult",
    }
    return mapping.get(category, "EnvelopeData")


def build_document() -> dict:
    model_paths: dict[str, dict] = {}
    for uri, category in server_uris():
        model_paths[uri] = {
            "post": {
                "summary": f"{category} inference",
                "tags": [category],
                "security": [{"bearerAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/ImgRequest"}}
                    },
                },
                "responses": {
                    "200": {
                        "description": "Success",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "allOf": [
                                        {"$ref": "#/components/schemas/Envelope"},
                                        {
                                            "properties": {
                                                "data": {"$ref": f"#/components/schemas/{data_schema_ref(category)}"}
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
                    "429": {"$ref": "#/components/responses/RateLimited"},
                    "500": {"$ref": "#/components/responses/InternalError"},
                    "504": {"$ref": "#/components/responses/GatewayTimeout"},
                },
            }
        }

    paths: dict[str, dict] = {
        "/healthz": {
            "get": {
                "summary": "Liveness probe",
                "tags": ["ops"],
                "responses": {
                    "200": {
                        "description": "OK",
                        "content": {
                            "application/json": {"schema": {"$ref": "#/components/schemas/Envelope"}}
                        },
                    }
                },
            }
        },
        "/ready": {
            "get": {
                "summary": "Readiness probe",
                "tags": ["ops"],
                "responses": {
                    "200": {
                        "description": "Ready",
                        "content": {
                            "application/json": {"schema": {"$ref": "#/components/schemas/Envelope"}}
                        },
                    },
                    "503": {"$ref": "#/components/responses/NotReady"},
                },
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

    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Mortred Model Server API",
            "version": "1.0.0",
            "description": (
                "Unified HTTP API for Mortred model servers. "
                "All model endpoints require `Authorization: Bearer <token>` "
                "when the server is configured with auth_token. "
                "Response envelope: {req_id, code, msg, data}; non-OK responses "
                "carry data:null."
            ),
        },
        "paths": paths,
        "components": {
            "securitySchemes": {
                "bearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "opaque"}
            },
            "schemas": {
                "Envelope": {
                    "type": "object",
                    "required": ["req_id", "code", "msg", "data"],
                    "properties": {
                        "req_id": {"type": "string", "description": "Client-provided or server-generated trace id"},
                        "code": {"type": "integer", "description": "Business status code (see docs/api-contract.md)"},
                        "msg": {"type": "string"},
                        "data": {"nullable": True, "description": "null on any non-OK response"},
                    },
                },
                "EnvelopeData": {"nullable": True},
                "ImgRequest": {
                    "type": "object",
                    "required": ["img_data"],
                    "properties": {
                        "img_data": {"type": "string", "description": "Base64 encoded image"},
                        "req_id": {"type": "string", "description": "Optional request trace id"},
                    },
                },
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
                        "image": {"type": "string", "description": "Base64 PNG segmentation mask"},
                        "colorized_mask": {"type": "string", "description": "Base64 PNG colorized mask"},
                    },
                },
                "MattingResult": {
                    "type": "object",
                    "required": ["image"],
                    "properties": {"image": {"type": "string", "description": "Base64 PNG matting result"}},
                },
                "EnhancementResult": {
                    "type": "object",
                    "required": ["image"],
                    "properties": {"image": {"type": "string", "description": "Base64 JPG enhanced image"}},
                },
                "DepthResult": {
                    "type": "object",
                    "required": ["image"],
                    "properties": {"image": {"type": "string", "description": "Base64 PNG colorized depth map"}},
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
            },
            "responses": {
                "BadRequest": {"description": "Malformed JSON or empty input", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Envelope"}}}},
                "Unauthorized": {"description": "Missing or invalid bearer token", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Envelope"}}}},
                "NotFound": {"description": "Unknown path", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Envelope"}}}},
                "MethodNotAllowed": {"description": "Only POST is allowed on model paths", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Envelope"}}}},
                "PayloadTooLarge": {"description": "Request body exceeds request_size_limit", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Envelope"}}}},
                "UnsupportedMediaType": {"description": "Content-Type is not application/json", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Envelope"}}}},
                "RateLimited": {"description": "Per-client-IP rate limit exceeded", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Envelope"}}}},
                "InternalError": {"description": "Model or server error", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Envelope"}}}},
                "NotReady": {"description": "Server is not ready", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Envelope"}}}},
                "GatewayTimeout": {"description": "Model run timeout", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Envelope"}}}},
            },
        },
    }


def render_header(json_text: str) -> str:
    return (
        "/************************************************\n"
        " * Author: Codex\n"
        " * File: openapi_doc.h\n"
        " *\n"
        " * Embedded OpenAPI document served at GET /openapi.json.\n"
        " * GENERATED FILE: do not edit by hand. Regenerate with:\n"
        " *   python scripts/gen_openapi.py\n"
        " * The content must stay byte-identical to docs/openapi.json\n"
        " * (enforced by scripts/check_consistency.py).\n"
        " ************************************************/\n"
        "\n"
        "#ifndef MORTRED_SERVER_OPENAPI_DOC_H\n"
        "#define MORTRED_SERVER_OPENAPI_DOC_H\n"
        "\n"
        "#include <string>\n"
        "\n"
        "namespace jinq {\n"
        "namespace server {\n"
        "\n"
        "inline const std::string k_openapi_doc_json = R\"MORTRED_OPENAPI(\n"
        + json_text
        + ")MORTRED_OPENAPI\";\n"
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
                print(f"ERROR: {p} (run: python scripts/gen_openapi.py)")
            return 1
        print("OpenAPI generation is up to date.")
        return 0

    OPENAPI_PATH.write_text(json_text, encoding="utf-8")
    HEADER_PATH.write_text(header_text, encoding="utf-8")
    print(f"wrote {OPENAPI_PATH.relative_to(ROOT)}")
    print(f"wrote {HEADER_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
