/************************************************
 * Author: Codex
 * File: openapi_doc.h
 *
 * Embedded OpenAPI document served at GET /openapi.json.
 * GENERATED FILE: do not edit by hand. Regenerate with:
 *   python scripts/gen_openapi.py
 * The content must stay byte-identical to docs/openapi.json
 * (enforced by scripts/check_consistency.py).
 ************************************************/

#ifndef MORTRED_SERVER_OPENAPI_DOC_H
#define MORTRED_SERVER_OPENAPI_DOC_H

#include <string>

namespace jinq {
namespace server {

inline const std::string k_openapi_doc_json = R"MORTRED_OPENAPI(
{
  "openapi": "3.0.0",
  "info": {
    "title": "Mortred Model Server API",
    "version": "1.0.0",
    "description": "Unified HTTP API for Mortred model servers. All model endpoints require `Authorization: Bearer <token>` when the server is configured with auth_token. Response envelope: {req_id, code, msg, data}; non-OK responses carry data:null."
  },
  "paths": {
    "/healthz": {
      "get": {
        "summary": "Liveness probe",
        "tags": [
          "ops"
        ],
        "responses": {
          "200": {
            "description": "OK",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Envelope"
                }
              }
            }
          }
        }
      }
    },
    "/ready": {
      "get": {
        "summary": "Readiness probe",
        "tags": [
          "ops"
        ],
        "responses": {
          "200": {
            "description": "Ready",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Envelope"
                }
              }
            }
          },
          "503": {
            "$ref": "#/components/responses/NotReady"
          }
        }
      }
    },
    "/metrics": {
      "get": {
        "summary": "Prometheus metrics",
        "tags": [
          "ops"
        ],
        "responses": {
          "200": {
            "description": "Prometheus text format",
            "content": {
              "text/plain": {
                "schema": {
                  "type": "string"
                }
              }
            }
          }
        }
      }
    },
    "/openapi.json": {
      "get": {
        "summary": "This OpenAPI document",
        "tags": [
          "ops"
        ],
        "responses": {
          "200": {
            "description": "OpenAPI 3.0 document",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object"
                }
              }
            }
          }
        }
      }
    },
    "/welcome": {
      "get": {
        "summary": "Legacy welcome page",
        "deprecated": true,
        "tags": [
          "legacy"
        ],
        "responses": {
          "200": {
            "description": "HTML page"
          }
        }
      }
    },
    "/hello_world": {
      "get": {
        "summary": "Legacy health check (kept for web console)",
        "deprecated": true,
        "tags": [
          "legacy"
        ],
        "responses": {
          "200": {
            "description": "HTML page"
          }
        }
      }
    },
    "/mortred_ai_server_v1/classification/densenet": {
      "post": {
        "summary": "classification inference",
        "tags": [
          "classification"
        ],
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ImgRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Envelope"
                    },
                    {
                      "properties": {
                        "data": {
                          "$ref": "#/components/schemas/ClassificationResult"
                        }
                      }
                    }
                  ]
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "404": {
            "$ref": "#/components/responses/NotFound"
          },
          "405": {
            "$ref": "#/components/responses/MethodNotAllowed"
          },
          "413": {
            "$ref": "#/components/responses/PayloadTooLarge"
          },
          "415": {
            "$ref": "#/components/responses/UnsupportedMediaType"
          },
          "429": {
            "$ref": "#/components/responses/RateLimited"
          },
          "500": {
            "$ref": "#/components/responses/InternalError"
          },
          "504": {
            "$ref": "#/components/responses/GatewayTimeout"
          }
        }
      }
    },
    "/mortred_ai_server_v1/classification/mobilenetv2": {
      "post": {
        "summary": "classification inference",
        "tags": [
          "classification"
        ],
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ImgRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Envelope"
                    },
                    {
                      "properties": {
                        "data": {
                          "$ref": "#/components/schemas/ClassificationResult"
                        }
                      }
                    }
                  ]
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "404": {
            "$ref": "#/components/responses/NotFound"
          },
          "405": {
            "$ref": "#/components/responses/MethodNotAllowed"
          },
          "413": {
            "$ref": "#/components/responses/PayloadTooLarge"
          },
          "415": {
            "$ref": "#/components/responses/UnsupportedMediaType"
          },
          "429": {
            "$ref": "#/components/responses/RateLimited"
          },
          "500": {
            "$ref": "#/components/responses/InternalError"
          },
          "504": {
            "$ref": "#/components/responses/GatewayTimeout"
          }
        }
      }
    },
    "/mortred_ai_server_v1/classification/resnet": {
      "post": {
        "summary": "classification inference",
        "tags": [
          "classification"
        ],
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ImgRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Envelope"
                    },
                    {
                      "properties": {
                        "data": {
                          "$ref": "#/components/schemas/ClassificationResult"
                        }
                      }
                    }
                  ]
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "404": {
            "$ref": "#/components/responses/NotFound"
          },
          "405": {
            "$ref": "#/components/responses/MethodNotAllowed"
          },
          "413": {
            "$ref": "#/components/responses/PayloadTooLarge"
          },
          "415": {
            "$ref": "#/components/responses/UnsupportedMediaType"
          },
          "429": {
            "$ref": "#/components/responses/RateLimited"
          },
          "500": {
            "$ref": "#/components/responses/InternalError"
          },
          "504": {
            "$ref": "#/components/responses/GatewayTimeout"
          }
        }
      }
    },
    "/mortred_ai_server_v1/enhancement/attentive_gan_derain": {
      "post": {
        "summary": "enhancement inference",
        "tags": [
          "enhancement"
        ],
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ImgRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Envelope"
                    },
                    {
                      "properties": {
                        "data": {
                          "$ref": "#/components/schemas/EnhancementResult"
                        }
                      }
                    }
                  ]
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "404": {
            "$ref": "#/components/responses/NotFound"
          },
          "405": {
            "$ref": "#/components/responses/MethodNotAllowed"
          },
          "413": {
            "$ref": "#/components/responses/PayloadTooLarge"
          },
          "415": {
            "$ref": "#/components/responses/UnsupportedMediaType"
          },
          "429": {
            "$ref": "#/components/responses/RateLimited"
          },
          "500": {
            "$ref": "#/components/responses/InternalError"
          },
          "504": {
            "$ref": "#/components/responses/GatewayTimeout"
          }
        }
      }
    },
    "/mortred_ai_server_v1/enhancement/enlighten_gan": {
      "post": {
        "summary": "enhancement inference",
        "tags": [
          "enhancement"
        ],
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ImgRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Envelope"
                    },
                    {
                      "properties": {
                        "data": {
                          "$ref": "#/components/schemas/EnhancementResult"
                        }
                      }
                    }
                  ]
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "404": {
            "$ref": "#/components/responses/NotFound"
          },
          "405": {
            "$ref": "#/components/responses/MethodNotAllowed"
          },
          "413": {
            "$ref": "#/components/responses/PayloadTooLarge"
          },
          "415": {
            "$ref": "#/components/responses/UnsupportedMediaType"
          },
          "429": {
            "$ref": "#/components/responses/RateLimited"
          },
          "500": {
            "$ref": "#/components/responses/InternalError"
          },
          "504": {
            "$ref": "#/components/responses/GatewayTimeout"
          }
        }
      }
    },
    "/mortred_ai_server_v1/enhancement/real_esrgan": {
      "post": {
        "summary": "enhancement inference",
        "tags": [
          "enhancement"
        ],
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ImgRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Envelope"
                    },
                    {
                      "properties": {
                        "data": {
                          "$ref": "#/components/schemas/EnhancementResult"
                        }
                      }
                    }
                  ]
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "404": {
            "$ref": "#/components/responses/NotFound"
          },
          "405": {
            "$ref": "#/components/responses/MethodNotAllowed"
          },
          "413": {
            "$ref": "#/components/responses/PayloadTooLarge"
          },
          "415": {
            "$ref": "#/components/responses/UnsupportedMediaType"
          },
          "429": {
            "$ref": "#/components/responses/RateLimited"
          },
          "500": {
            "$ref": "#/components/responses/InternalError"
          },
          "504": {
            "$ref": "#/components/responses/GatewayTimeout"
          }
        }
      }
    },
    "/mortred_ai_server_v1/feature_point/superpoint": {
      "post": {
        "summary": "feature_point inference",
        "tags": [
          "feature_point"
        ],
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ImgRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Envelope"
                    },
                    {
                      "properties": {
                        "data": {
                          "$ref": "#/components/schemas/FeaturePointResult"
                        }
                      }
                    }
                  ]
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "404": {
            "$ref": "#/components/responses/NotFound"
          },
          "405": {
            "$ref": "#/components/responses/MethodNotAllowed"
          },
          "413": {
            "$ref": "#/components/responses/PayloadTooLarge"
          },
          "415": {
            "$ref": "#/components/responses/UnsupportedMediaType"
          },
          "429": {
            "$ref": "#/components/responses/RateLimited"
          },
          "500": {
            "$ref": "#/components/responses/InternalError"
          },
          "504": {
            "$ref": "#/components/responses/GatewayTimeout"
          }
        }
      }
    },
    "/mortred_ai_server_v1/matting/modnet": {
      "post": {
        "summary": "matting inference",
        "tags": [
          "matting"
        ],
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ImgRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Envelope"
                    },
                    {
                      "properties": {
                        "data": {
                          "$ref": "#/components/schemas/MattingResult"
                        }
                      }
                    }
                  ]
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "404": {
            "$ref": "#/components/responses/NotFound"
          },
          "405": {
            "$ref": "#/components/responses/MethodNotAllowed"
          },
          "413": {
            "$ref": "#/components/responses/PayloadTooLarge"
          },
          "415": {
            "$ref": "#/components/responses/UnsupportedMediaType"
          },
          "429": {
            "$ref": "#/components/responses/RateLimited"
          },
          "500": {
            "$ref": "#/components/responses/InternalError"
          },
          "504": {
            "$ref": "#/components/responses/GatewayTimeout"
          }
        }
      }
    },
    "/mortred_ai_server_v1/matting/pp_matting": {
      "post": {
        "summary": "matting inference",
        "tags": [
          "matting"
        ],
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ImgRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Envelope"
                    },
                    {
                      "properties": {
                        "data": {
                          "$ref": "#/components/schemas/MattingResult"
                        }
                      }
                    }
                  ]
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "404": {
            "$ref": "#/components/responses/NotFound"
          },
          "405": {
            "$ref": "#/components/responses/MethodNotAllowed"
          },
          "413": {
            "$ref": "#/components/responses/PayloadTooLarge"
          },
          "415": {
            "$ref": "#/components/responses/UnsupportedMediaType"
          },
          "429": {
            "$ref": "#/components/responses/RateLimited"
          },
          "500": {
            "$ref": "#/components/responses/InternalError"
          },
          "504": {
            "$ref": "#/components/responses/GatewayTimeout"
          }
        }
      }
    },
    "/mortred_ai_server_v1/mono_depth_estimation/depth_anything": {
      "post": {
        "summary": "mono_depth_estimation inference",
        "tags": [
          "mono_depth_estimation"
        ],
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ImgRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Envelope"
                    },
                    {
                      "properties": {
                        "data": {
                          "$ref": "#/components/schemas/DepthResult"
                        }
                      }
                    }
                  ]
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "404": {
            "$ref": "#/components/responses/NotFound"
          },
          "405": {
            "$ref": "#/components/responses/MethodNotAllowed"
          },
          "413": {
            "$ref": "#/components/responses/PayloadTooLarge"
          },
          "415": {
            "$ref": "#/components/responses/UnsupportedMediaType"
          },
          "429": {
            "$ref": "#/components/responses/RateLimited"
          },
          "500": {
            "$ref": "#/components/responses/InternalError"
          },
          "504": {
            "$ref": "#/components/responses/GatewayTimeout"
          }
        }
      }
    },
    "/mortred_ai_server_v1/mono_depth_estimation/metric3d": {
      "post": {
        "summary": "mono_depth_estimation inference",
        "tags": [
          "mono_depth_estimation"
        ],
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ImgRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Envelope"
                    },
                    {
                      "properties": {
                        "data": {
                          "$ref": "#/components/schemas/DepthResult"
                        }
                      }
                    }
                  ]
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "404": {
            "$ref": "#/components/responses/NotFound"
          },
          "405": {
            "$ref": "#/components/responses/MethodNotAllowed"
          },
          "413": {
            "$ref": "#/components/responses/PayloadTooLarge"
          },
          "415": {
            "$ref": "#/components/responses/UnsupportedMediaType"
          },
          "429": {
            "$ref": "#/components/responses/RateLimited"
          },
          "500": {
            "$ref": "#/components/responses/InternalError"
          },
          "504": {
            "$ref": "#/components/responses/GatewayTimeout"
          }
        }
      }
    },
    "/mortred_ai_server_v1/obj_detection/center_face": {
      "post": {
        "summary": "object_detection inference",
        "tags": [
          "object_detection"
        ],
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ImgRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Envelope"
                    },
                    {
                      "properties": {
                        "data": {
                          "$ref": "#/components/schemas/DetectionResult"
                        }
                      }
                    }
                  ]
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "404": {
            "$ref": "#/components/responses/NotFound"
          },
          "405": {
            "$ref": "#/components/responses/MethodNotAllowed"
          },
          "413": {
            "$ref": "#/components/responses/PayloadTooLarge"
          },
          "415": {
            "$ref": "#/components/responses/UnsupportedMediaType"
          },
          "429": {
            "$ref": "#/components/responses/RateLimited"
          },
          "500": {
            "$ref": "#/components/responses/InternalError"
          },
          "504": {
            "$ref": "#/components/responses/GatewayTimeout"
          }
        }
      }
    },
    "/mortred_ai_server_v1/obj_detection/libface": {
      "post": {
        "summary": "object_detection inference",
        "tags": [
          "object_detection"
        ],
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ImgRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Envelope"
                    },
                    {
                      "properties": {
                        "data": {
                          "$ref": "#/components/schemas/DetectionResult"
                        }
                      }
                    }
                  ]
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "404": {
            "$ref": "#/components/responses/NotFound"
          },
          "405": {
            "$ref": "#/components/responses/MethodNotAllowed"
          },
          "413": {
            "$ref": "#/components/responses/PayloadTooLarge"
          },
          "415": {
            "$ref": "#/components/responses/UnsupportedMediaType"
          },
          "429": {
            "$ref": "#/components/responses/RateLimited"
          },
          "500": {
            "$ref": "#/components/responses/InternalError"
          },
          "504": {
            "$ref": "#/components/responses/GatewayTimeout"
          }
        }
      }
    },
    "/mortred_ai_server_v1/obj_detection/nanodet": {
      "post": {
        "summary": "object_detection inference",
        "tags": [
          "object_detection"
        ],
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ImgRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Envelope"
                    },
                    {
                      "properties": {
                        "data": {
                          "$ref": "#/components/schemas/DetectionResult"
                        }
                      }
                    }
                  ]
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "404": {
            "$ref": "#/components/responses/NotFound"
          },
          "405": {
            "$ref": "#/components/responses/MethodNotAllowed"
          },
          "413": {
            "$ref": "#/components/responses/PayloadTooLarge"
          },
          "415": {
            "$ref": "#/components/responses/UnsupportedMediaType"
          },
          "429": {
            "$ref": "#/components/responses/RateLimited"
          },
          "500": {
            "$ref": "#/components/responses/InternalError"
          },
          "504": {
            "$ref": "#/components/responses/GatewayTimeout"
          }
        }
      }
    },
    "/mortred_ai_server_v1/obj_detection/yolov5": {
      "post": {
        "summary": "object_detection inference",
        "tags": [
          "object_detection"
        ],
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ImgRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Envelope"
                    },
                    {
                      "properties": {
                        "data": {
                          "$ref": "#/components/schemas/DetectionResult"
                        }
                      }
                    }
                  ]
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "404": {
            "$ref": "#/components/responses/NotFound"
          },
          "405": {
            "$ref": "#/components/responses/MethodNotAllowed"
          },
          "413": {
            "$ref": "#/components/responses/PayloadTooLarge"
          },
          "415": {
            "$ref": "#/components/responses/UnsupportedMediaType"
          },
          "429": {
            "$ref": "#/components/responses/RateLimited"
          },
          "500": {
            "$ref": "#/components/responses/InternalError"
          },
          "504": {
            "$ref": "#/components/responses/GatewayTimeout"
          }
        }
      }
    },
    "/mortred_ai_server_v1/obj_detection/yolov6": {
      "post": {
        "summary": "object_detection inference",
        "tags": [
          "object_detection"
        ],
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ImgRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Envelope"
                    },
                    {
                      "properties": {
                        "data": {
                          "$ref": "#/components/schemas/DetectionResult"
                        }
                      }
                    }
                  ]
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "404": {
            "$ref": "#/components/responses/NotFound"
          },
          "405": {
            "$ref": "#/components/responses/MethodNotAllowed"
          },
          "413": {
            "$ref": "#/components/responses/PayloadTooLarge"
          },
          "415": {
            "$ref": "#/components/responses/UnsupportedMediaType"
          },
          "429": {
            "$ref": "#/components/responses/RateLimited"
          },
          "500": {
            "$ref": "#/components/responses/InternalError"
          },
          "504": {
            "$ref": "#/components/responses/GatewayTimeout"
          }
        }
      }
    },
    "/mortred_ai_server_v1/obj_detection/yolov7": {
      "post": {
        "summary": "object_detection inference",
        "tags": [
          "object_detection"
        ],
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ImgRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Envelope"
                    },
                    {
                      "properties": {
                        "data": {
                          "$ref": "#/components/schemas/DetectionResult"
                        }
                      }
                    }
                  ]
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "404": {
            "$ref": "#/components/responses/NotFound"
          },
          "405": {
            "$ref": "#/components/responses/MethodNotAllowed"
          },
          "413": {
            "$ref": "#/components/responses/PayloadTooLarge"
          },
          "415": {
            "$ref": "#/components/responses/UnsupportedMediaType"
          },
          "429": {
            "$ref": "#/components/responses/RateLimited"
          },
          "500": {
            "$ref": "#/components/responses/InternalError"
          },
          "504": {
            "$ref": "#/components/responses/GatewayTimeout"
          }
        }
      }
    },
    "/mortred_ai_server_v1/obj_detection/yolov8": {
      "post": {
        "summary": "object_detection inference",
        "tags": [
          "object_detection"
        ],
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ImgRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Envelope"
                    },
                    {
                      "properties": {
                        "data": {
                          "$ref": "#/components/schemas/DetectionResult"
                        }
                      }
                    }
                  ]
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "404": {
            "$ref": "#/components/responses/NotFound"
          },
          "405": {
            "$ref": "#/components/responses/MethodNotAllowed"
          },
          "413": {
            "$ref": "#/components/responses/PayloadTooLarge"
          },
          "415": {
            "$ref": "#/components/responses/UnsupportedMediaType"
          },
          "429": {
            "$ref": "#/components/responses/RateLimited"
          },
          "500": {
            "$ref": "#/components/responses/InternalError"
          },
          "504": {
            "$ref": "#/components/responses/GatewayTimeout"
          }
        }
      }
    },
    "/mortred_ai_server_v1/ocr/dbtext": {
      "post": {
        "summary": "ocr inference",
        "tags": [
          "ocr"
        ],
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ImgRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Envelope"
                    },
                    {
                      "properties": {
                        "data": {
                          "$ref": "#/components/schemas/TextRegionResult"
                        }
                      }
                    }
                  ]
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "404": {
            "$ref": "#/components/responses/NotFound"
          },
          "405": {
            "$ref": "#/components/responses/MethodNotAllowed"
          },
          "413": {
            "$ref": "#/components/responses/PayloadTooLarge"
          },
          "415": {
            "$ref": "#/components/responses/UnsupportedMediaType"
          },
          "429": {
            "$ref": "#/components/responses/RateLimited"
          },
          "500": {
            "$ref": "#/components/responses/InternalError"
          },
          "504": {
            "$ref": "#/components/responses/GatewayTimeout"
          }
        }
      }
    },
    "/mortred_ai_server_v1/scene_segmentation/bisenetv2": {
      "post": {
        "summary": "scene_segmentation inference",
        "tags": [
          "scene_segmentation"
        ],
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ImgRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Envelope"
                    },
                    {
                      "properties": {
                        "data": {
                          "$ref": "#/components/schemas/SegmentationResult"
                        }
                      }
                    }
                  ]
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "404": {
            "$ref": "#/components/responses/NotFound"
          },
          "405": {
            "$ref": "#/components/responses/MethodNotAllowed"
          },
          "413": {
            "$ref": "#/components/responses/PayloadTooLarge"
          },
          "415": {
            "$ref": "#/components/responses/UnsupportedMediaType"
          },
          "429": {
            "$ref": "#/components/responses/RateLimited"
          },
          "500": {
            "$ref": "#/components/responses/InternalError"
          },
          "504": {
            "$ref": "#/components/responses/GatewayTimeout"
          }
        }
      }
    },
    "/mortred_ai_server_v1/scene_segmentation/hrnet": {
      "post": {
        "summary": "scene_segmentation inference",
        "tags": [
          "scene_segmentation"
        ],
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ImgRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Envelope"
                    },
                    {
                      "properties": {
                        "data": {
                          "$ref": "#/components/schemas/SegmentationResult"
                        }
                      }
                    }
                  ]
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "404": {
            "$ref": "#/components/responses/NotFound"
          },
          "405": {
            "$ref": "#/components/responses/MethodNotAllowed"
          },
          "413": {
            "$ref": "#/components/responses/PayloadTooLarge"
          },
          "415": {
            "$ref": "#/components/responses/UnsupportedMediaType"
          },
          "429": {
            "$ref": "#/components/responses/RateLimited"
          },
          "500": {
            "$ref": "#/components/responses/InternalError"
          },
          "504": {
            "$ref": "#/components/responses/GatewayTimeout"
          }
        }
      }
    },
    "/mortred_ai_server_v1/scene_segmentation/pphuman_seg": {
      "post": {
        "summary": "scene_segmentation inference",
        "tags": [
          "scene_segmentation"
        ],
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ImgRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Envelope"
                    },
                    {
                      "properties": {
                        "data": {
                          "$ref": "#/components/schemas/SegmentationResult"
                        }
                      }
                    }
                  ]
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "404": {
            "$ref": "#/components/responses/NotFound"
          },
          "405": {
            "$ref": "#/components/responses/MethodNotAllowed"
          },
          "413": {
            "$ref": "#/components/responses/PayloadTooLarge"
          },
          "415": {
            "$ref": "#/components/responses/UnsupportedMediaType"
          },
          "429": {
            "$ref": "#/components/responses/RateLimited"
          },
          "500": {
            "$ref": "#/components/responses/InternalError"
          },
          "504": {
            "$ref": "#/components/responses/GatewayTimeout"
          }
        }
      }
    }
  },
  "components": {
    "securitySchemes": {
      "bearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "opaque"
      }
    },
    "schemas": {
      "Envelope": {
        "type": "object",
        "required": [
          "req_id",
          "code",
          "msg",
          "data"
        ],
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Client-provided or server-generated trace id"
          },
          "code": {
            "type": "integer",
            "description": "Business status code (see docs/api-contract.md)"
          },
          "msg": {
            "type": "string"
          },
          "data": {
            "nullable": true,
            "description": "null on any non-OK response"
          }
        }
      },
      "EnvelopeData": {
        "nullable": true
      },
      "ImgRequest": {
        "type": "object",
        "required": [
          "img_data"
        ],
        "properties": {
          "img_data": {
            "type": "string",
            "description": "Base64 encoded image"
          },
          "req_id": {
            "type": "string",
            "description": "Optional request trace id"
          }
        }
      },
      "ClassificationResult": {
        "type": "object",
        "required": [
          "class_id",
          "category",
          "scores"
        ],
        "properties": {
          "class_id": {
            "type": "integer"
          },
          "category": {
            "type": "string"
          },
          "scores": {
            "type": "array",
            "items": {
              "type": "number"
            }
          }
        }
      },
      "BBox": {
        "type": "array",
        "items": {
          "type": "number"
        },
        "minItems": 4,
        "maxItems": 4,
        "description": "[x1, y1, x2, y2]"
      },
      "DetectionItem": {
        "type": "object",
        "required": [
          "class_id",
          "score",
          "category",
          "bbox",
          "detail_infos"
        ],
        "properties": {
          "class_id": {
            "type": "integer"
          },
          "score": {
            "type": "number"
          },
          "category": {
            "type": "string"
          },
          "bbox": {
            "$ref": "#/components/schemas/BBox"
          },
          "detail_infos": {
            "type": "object"
          }
        }
      },
      "DetectionResult": {
        "type": "array",
        "items": {
          "$ref": "#/components/schemas/DetectionItem"
        }
      },
      "FaceItem": {
        "type": "object",
        "required": [
          "class_id",
          "score",
          "category",
          "bbox",
          "landmarks",
          "detail_infos"
        ],
        "properties": {
          "class_id": {
            "type": "integer"
          },
          "score": {
            "type": "number"
          },
          "category": {
            "type": "string"
          },
          "bbox": {
            "$ref": "#/components/schemas/BBox"
          },
          "landmarks": {
            "type": "array",
            "items": {
              "type": "array",
              "items": {
                "type": "number"
              },
              "minItems": 2,
              "maxItems": 2
            }
          },
          "detail_infos": {
            "type": "object"
          }
        }
      },
      "FaceResult": {
        "type": "array",
        "items": {
          "$ref": "#/components/schemas/FaceItem"
        }
      },
      "TextRegionItem": {
        "type": "object",
        "required": [
          "score",
          "bbox",
          "polygon",
          "detail_infos"
        ],
        "properties": {
          "score": {
            "type": "number"
          },
          "bbox": {
            "$ref": "#/components/schemas/BBox"
          },
          "polygon": {
            "type": "array",
            "items": {
              "type": "array",
              "items": {
                "type": "number"
              },
              "minItems": 2,
              "maxItems": 2
            }
          },
          "detail_infos": {
            "type": "object"
          }
        }
      },
      "TextRegionResult": {
        "type": "array",
        "items": {
          "$ref": "#/components/schemas/TextRegionItem"
        }
      },
      "SegmentationResult": {
        "type": "object",
        "required": [
          "image",
          "colorized_mask"
        ],
        "properties": {
          "image": {
            "type": "string",
            "description": "Base64 PNG segmentation mask"
          },
          "colorized_mask": {
            "type": "string",
            "description": "Base64 PNG colorized mask"
          }
        }
      },
      "MattingResult": {
        "type": "object",
        "required": [
          "image"
        ],
        "properties": {
          "image": {
            "type": "string",
            "description": "Base64 PNG matting result"
          }
        }
      },
      "EnhancementResult": {
        "type": "object",
        "required": [
          "image"
        ],
        "properties": {
          "image": {
            "type": "string",
            "description": "Base64 JPG enhanced image"
          }
        }
      },
      "DepthResult": {
        "type": "object",
        "required": [
          "image"
        ],
        "properties": {
          "image": {
            "type": "string",
            "description": "Base64 PNG colorized depth map"
          }
        }
      },
      "FeaturePointItem": {
        "type": "object",
        "required": [
          "score",
          "location",
          "descriptor"
        ],
        "properties": {
          "score": {
            "type": "number"
          },
          "location": {
            "type": "array",
            "items": {
              "type": "number"
            },
            "minItems": 2,
            "maxItems": 2
          },
          "descriptor": {
            "type": "array",
            "items": {
              "type": "number"
            }
          }
        }
      },
      "FeaturePointResult": {
        "type": "array",
        "items": {
          "$ref": "#/components/schemas/FeaturePointItem"
        }
      }
    },
    "responses": {
      "BadRequest": {
        "description": "Malformed JSON or empty input",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/Envelope"
            }
          }
        }
      },
      "Unauthorized": {
        "description": "Missing or invalid bearer token",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/Envelope"
            }
          }
        }
      },
      "NotFound": {
        "description": "Unknown path",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/Envelope"
            }
          }
        }
      },
      "MethodNotAllowed": {
        "description": "Only POST is allowed on model paths",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/Envelope"
            }
          }
        }
      },
      "PayloadTooLarge": {
        "description": "Request body exceeds request_size_limit",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/Envelope"
            }
          }
        }
      },
      "UnsupportedMediaType": {
        "description": "Content-Type is not application/json",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/Envelope"
            }
          }
        }
      },
      "RateLimited": {
        "description": "Per-client-IP rate limit exceeded",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/Envelope"
            }
          }
        }
      },
      "InternalError": {
        "description": "Model or server error",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/Envelope"
            }
          }
        }
      },
      "NotReady": {
        "description": "Server is not ready",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/Envelope"
            }
          }
        }
      },
      "GatewayTimeout": {
        "description": "Model run timeout",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/Envelope"
            }
          }
        }
      }
    }
  }
}
)MORTRED_OPENAPI";

}  // namespace server
}  // namespace jinq

#endif  // MORTRED_SERVER_OPENAPI_DOC_H
