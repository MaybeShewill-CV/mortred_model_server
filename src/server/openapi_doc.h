/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: openapi_doc.h
* Date: 26-8-31
************************************************/

// GENERATED FILE: do not edit by hand. Regenerate with:
//   cmake --build <full-build-dir> --target contract_dump
//   <full-build-dir>/bin/contract_dump > docs/contract_dump.json
//   python scripts/gen_openapi.py

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
    "description": "Unified HTTP API for Mortred model servers. Model endpoints require `Authorization: Bearer <token>` when auth_token is configured. Request envelope: {req_id, images[], params, options}; response envelope: {status, status_str, task_id, model, results[], server_time_ms, partial}. The legacy img_data field was removed: it answers 422 with a migration hint.",
    "x-contract-hash": "8a02ce10bce3932d07173fadf0a5fb16e99d8f0f70d12cc5befe5c752e5b3a5b"
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
            "description": "OK"
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
            "description": "Ready"
          },
          "503": {
            "description": "Not ready"
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
    "/mortred_ai_server_v1/enhancement/attentive_gan_derain": {
      "post": {
        "summary": "attentive gan derain",
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
                "$ref": "#/components/schemas/Request_ATTENTIVE_GAN_DERAIN_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
        "summary": "bisenetv2 segmentation",
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
                "$ref": "#/components/schemas/Request_BISENETV2_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
        "summary": "center face detection",
        "tags": [
          "face_detection"
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
                "$ref": "#/components/schemas/Request_CENTER_FACE_DETECTION_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
                              },
                              {
                                "properties": {
                                  "data": {
                                    "$ref": "#/components/schemas/FaceResult"
                                  }
                                }
                              }
                            ]
                          }
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
    "/mortred_ai_server_v1/diffusion/cls_cond_ddim": {
      "post": {
        "summary": "class conditional DDIM sampler",
        "tags": [
          "diffusion"
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
                "$ref": "#/components/schemas/Request_CLS_COND_DDIM_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
                              },
                              {
                                "properties": {
                                  "data": {
                                    "$ref": "#/components/schemas/ImageResult"
                                  }
                                }
                              }
                            ]
                          }
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
        "summary": "dbnet",
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
                "$ref": "#/components/schemas/Request_DBNET_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
    "/mortred_ai_server_v1/diffusion/ddim": {
      "post": {
        "summary": "DDIM diffusion sampler",
        "tags": [
          "diffusion"
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
                "$ref": "#/components/schemas/Request_DDIM_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
                              },
                              {
                                "properties": {
                                  "data": {
                                    "$ref": "#/components/schemas/ImageResult"
                                  }
                                }
                              }
                            ]
                          }
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
    "/mortred_ai_server_v1/diffusion/ddpm": {
      "post": {
        "summary": "DDPM diffusion sampler",
        "tags": [
          "diffusion"
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
                "$ref": "#/components/schemas/Request_DDPM_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
                              },
                              {
                                "properties": {
                                  "data": {
                                    "$ref": "#/components/schemas/ImageResult"
                                  }
                                }
                              }
                            ]
                          }
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
    "/mortred_ai_server_v1/classification/densenet": {
      "post": {
        "summary": "densenet classification",
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
                "$ref": "#/components/schemas/Request_DENSENET_CLASSIFICATION_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
        "summary": "depth anything estimation",
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
                "$ref": "#/components/schemas/Request_DEPTH_ANYTHING_ESTIMATION_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
        "summary": "enlighten gan",
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
                "$ref": "#/components/schemas/Request_ENLIGHTEN_GAN_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
        "summary": "hrnet segmentation",
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
                "$ref": "#/components/schemas/Request_HRNET_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
    "/mortred_ai_server_v1/diffusion/ldm": {
      "post": {
        "summary": "latent diffusion sampler",
        "tags": [
          "diffusion"
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
                "$ref": "#/components/schemas/Request_LDM_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
                              },
                              {
                                "properties": {
                                  "data": {
                                    "$ref": "#/components/schemas/ImageResult"
                                  }
                                }
                              }
                            ]
                          }
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
        "summary": "libface face detection",
        "tags": [
          "face_detection"
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
                "$ref": "#/components/schemas/Request_LIBFACE_DETECTION_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
                              },
                              {
                                "properties": {
                                  "data": {
                                    "$ref": "#/components/schemas/FaceResult"
                                  }
                                }
                              }
                            ]
                          }
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
        "summary": "metric3d estimation",
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
                "$ref": "#/components/schemas/Request_METRIC3D_ESTIMATION_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
        "summary": "Mobilenetv2 classification",
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
                "$ref": "#/components/schemas/Request_MOBILENETV2_CLASSIFICATION_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
        "summary": "modnet",
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
                "$ref": "#/components/schemas/Request_MODNET_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
        "summary": "nanodet object detection",
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
                "$ref": "#/components/schemas/Request_NANODET_DETECTION_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
        "summary": "pphuman segmentation",
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
                "$ref": "#/components/schemas/Request_PPHUMAN_SEG_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
        "summary": "pp matting",
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
                "$ref": "#/components/schemas/Request_PP_MATTING_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
        "summary": "real esr-gan",
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
                "$ref": "#/components/schemas/Request_REAL_ESRGAN_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
        "summary": "Resnet classification",
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
                "$ref": "#/components/schemas/Request_RESNET_CLASSIFICATION_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
    "/mortred_ai_server_v1/sam/amg": {
      "post": {
        "summary": "SAM automatic mask generator",
        "tags": [
          "segment_anything_amg"
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
                "$ref": "#/components/schemas/Request_SAM_AMG_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
                              },
                              {
                                "properties": {
                                  "data": {
                                    "$ref": "#/components/schemas/SamAmgResult"
                                  }
                                }
                              }
                            ]
                          }
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
        "summary": "Superpoint feature point detection",
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
                "$ref": "#/components/schemas/Request_SUPERPOINT_FP_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
        "summary": "Yolov5 object detection",
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
                "$ref": "#/components/schemas/Request_YOLOV5_DETECTION_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
        "summary": "Yolov6 object detection",
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
                "$ref": "#/components/schemas/Request_YOLOV6_DETECTION_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
        "summary": "Yolov7 object detection",
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
                "$ref": "#/components/schemas/Request_YOLOV7_DETECTION_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
        "summary": "Yolov8 object detection",
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
                "$ref": "#/components/schemas/Request_YOLOV8_DETECTION_SERVER"
              }
            },
            "image/png": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "image/jpeg": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            },
            "application/octet-stream": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          }
        },
        "parameters": [
          {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Trace id (raw encoding only; JSON uses req_id)"
          },
          {
            "name": "X-Mortred-Params",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"score_threshold\":0.35}"
          },
          {
            "name": "X-Mortred-Options",
            "in": "header",
            "required": false,
            "schema": {
              "type": "string"
            },
            "description": "Compact JSON object, e.g. {\"encoding\":\"jpeg\"}"
          }
        ],
        "responses": {
          "200": {
            "description": "Unified envelope; results[] aligns with images[]. A mid-request deadline returns the completed items with partial=true.",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/UnifiedResponse"
                    },
                    {
                      "properties": {
                        "results": {
                          "type": "array",
                          "items": {
                            "allOf": [
                              {
                                "$ref": "#/components/schemas/ResponseItem"
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
          "422": {
            "$ref": "#/components/responses/ValidationError"
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
      "Request_ATTENTIVE_GAN_DERAIN_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "type": "object",
            "properties": {},
            "additionalProperties": false,
            "description": "This model declares no request-level parameters"
          }
        }
      },
      "Request_BISENETV2_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "type": "object",
            "properties": {},
            "additionalProperties": false,
            "description": "This model declares no request-level parameters"
          }
        }
      },
      "Request_CENTER_FACE_DETECTION_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "$ref": "#/components/schemas/Params_CENTER_FACE_DETECTION_SERVER"
          }
        }
      },
      "Params_CENTER_FACE_DETECTION_SERVER": {
        "type": "object",
        "properties": {
          "score_threshold": {
            "type": "number",
            "format": "float",
            "description": "confidence threshold",
            "minimum": 0.0,
            "maximum": 1.0
          },
          "nms_threshold": {
            "type": "number",
            "format": "float",
            "description": "per-class NMS IoU threshold",
            "minimum": 0.1,
            "maximum": 1.0
          },
          "top_k": {
            "type": "integer",
            "format": "int32",
            "description": "keep at most k detections",
            "minimum": 1.0,
            "maximum": 10000.0
          }
        },
        "additionalProperties": false,
        "description": "Request-level parameter overrides (strict: unknown keys -> 422)"
      },
      "Request_CLS_COND_DDIM_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "$ref": "#/components/schemas/Params_CLS_COND_DDIM_SERVER"
          }
        }
      },
      "Params_CLS_COND_DDIM_SERVER": {
        "type": "object",
        "properties": {
          "sample_steps": {
            "type": "integer",
            "format": "int32",
            "description": "DDIM sampling steps (fewer = faster)",
            "minimum": 1.0,
            "maximum": 1000.0
          },
          "eta": {
            "type": "number",
            "format": "float",
            "description": "stochasticity (0 = deterministic ODE, 1 = full stochastic)",
            "minimum": 0.0,
            "maximum": 1.0
          },
          "cls_id": {
            "type": "integer",
            "format": "int32",
            "description": "conditioning class id (model dependent)",
            "minimum": 0.0,
            "maximum": 9999.0
          }
        },
        "additionalProperties": false,
        "description": "Request-level parameter overrides (strict: unknown keys -> 422)"
      },
      "Request_DBNET_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "$ref": "#/components/schemas/Params_DBNET_SERVER"
          }
        }
      },
      "Params_DBNET_SERVER": {
        "type": "object",
        "properties": {
          "score_threshold": {
            "type": "number",
            "format": "float",
            "description": "text region confidence threshold",
            "minimum": 0.1,
            "maximum": 0.9
          },
          "top_k": {
            "type": "integer",
            "format": "int32",
            "description": "keep at most k text regions",
            "minimum": 1.0,
            "maximum": 10000.0
          }
        },
        "additionalProperties": false,
        "description": "Request-level parameter overrides (strict: unknown keys -> 422)"
      },
      "Request_DDIM_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "$ref": "#/components/schemas/Params_DDIM_SERVER"
          }
        }
      },
      "Params_DDIM_SERVER": {
        "type": "object",
        "properties": {
          "sample_steps": {
            "type": "integer",
            "format": "int32",
            "description": "DDIM sampling steps (fewer = faster)",
            "minimum": 1.0,
            "maximum": 1000.0
          },
          "eta": {
            "type": "number",
            "format": "float",
            "description": "stochasticity (0 = deterministic ODE, 1 = full stochastic)",
            "minimum": 0.0,
            "maximum": 1.0
          }
        },
        "additionalProperties": false,
        "description": "Request-level parameter overrides (strict: unknown keys -> 422)"
      },
      "Request_DDPM_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "$ref": "#/components/schemas/Params_DDPM_SERVER"
          }
        }
      },
      "Params_DDPM_SERVER": {
        "type": "object",
        "properties": {
          "timesteps": {
            "type": "integer",
            "format": "int32",
            "description": "sampling timesteps (more = slower, higher quality)",
            "minimum": 1.0,
            "maximum": 1000.0
          }
        },
        "additionalProperties": false,
        "description": "Request-level parameter overrides (strict: unknown keys -> 422)"
      },
      "Request_DENSENET_CLASSIFICATION_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "$ref": "#/components/schemas/Params_DENSENET_CLASSIFICATION_SERVER"
          }
        }
      },
      "Params_DENSENET_CLASSIFICATION_SERVER": {
        "type": "object",
        "properties": {
          "top_k": {
            "type": "integer",
            "format": "int32",
            "description": "keep the k highest scores, descending",
            "minimum": 1.0,
            "maximum": 1000.0
          }
        },
        "additionalProperties": false,
        "description": "Request-level parameter overrides (strict: unknown keys -> 422)"
      },
      "Request_DEPTH_ANYTHING_ESTIMATION_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "type": "object",
            "properties": {},
            "additionalProperties": false,
            "description": "This model declares no request-level parameters"
          }
        }
      },
      "Request_ENLIGHTEN_GAN_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "type": "object",
            "properties": {},
            "additionalProperties": false,
            "description": "This model declares no request-level parameters"
          }
        }
      },
      "Request_HRNET_SEGMENTATION_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "type": "object",
            "properties": {},
            "additionalProperties": false,
            "description": "This model declares no request-level parameters"
          }
        }
      },
      "Request_HRNET_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "type": "object",
            "properties": {},
            "additionalProperties": false,
            "description": "This model declares no request-level parameters"
          }
        }
      },
      "Request_LDM_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "$ref": "#/components/schemas/Params_LDM_SERVER"
          }
        }
      },
      "Params_LDM_SERVER": {
        "type": "object",
        "properties": {
          "step_size": {
            "type": "integer",
            "format": "int32",
            "description": "latent sampler steps (fewer = faster)",
            "minimum": 1.0,
            "maximum": 1000.0
          }
        },
        "additionalProperties": false,
        "description": "Request-level parameter overrides (strict: unknown keys -> 422)"
      },
      "Request_LIBFACE_DETECTION_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "$ref": "#/components/schemas/Params_LIBFACE_DETECTION_SERVER"
          }
        }
      },
      "Params_LIBFACE_DETECTION_SERVER": {
        "type": "object",
        "properties": {
          "score_threshold": {
            "type": "number",
            "format": "float",
            "description": "confidence threshold",
            "minimum": 0.0,
            "maximum": 1.0
          },
          "nms_threshold": {
            "type": "number",
            "format": "float",
            "description": "per-class NMS IoU threshold",
            "minimum": 0.1,
            "maximum": 1.0
          },
          "top_k": {
            "type": "integer",
            "format": "int32",
            "description": "keep at most k detections",
            "minimum": 1.0,
            "maximum": 10000.0
          }
        },
        "additionalProperties": false,
        "description": "Request-level parameter overrides (strict: unknown keys -> 422)"
      },
      "Request_METRIC3D_ESTIMATION_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "type": "object",
            "properties": {},
            "additionalProperties": false,
            "description": "This model declares no request-level parameters"
          }
        }
      },
      "Request_MOBILENETV2_CLASSIFICATION_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "$ref": "#/components/schemas/Params_MOBILENETV2_CLASSIFICATION_SERVER"
          }
        }
      },
      "Params_MOBILENETV2_CLASSIFICATION_SERVER": {
        "type": "object",
        "properties": {
          "top_k": {
            "type": "integer",
            "format": "int32",
            "description": "keep the k highest scores, descending",
            "minimum": 1.0,
            "maximum": 1000.0
          }
        },
        "additionalProperties": false,
        "description": "Request-level parameter overrides (strict: unknown keys -> 422)"
      },
      "Request_MODNET_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "type": "object",
            "properties": {},
            "additionalProperties": false,
            "description": "This model declares no request-level parameters"
          }
        }
      },
      "Request_NANODET_DETECTION_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "$ref": "#/components/schemas/Params_NANODET_DETECTION_SERVER"
          }
        }
      },
      "Params_NANODET_DETECTION_SERVER": {
        "type": "object",
        "properties": {
          "score_threshold": {
            "type": "number",
            "format": "float",
            "description": "confidence threshold",
            "minimum": 0.0,
            "maximum": 1.0
          },
          "nms_threshold": {
            "type": "number",
            "format": "float",
            "description": "per-class NMS IoU threshold",
            "minimum": 0.1,
            "maximum": 1.0
          },
          "top_k": {
            "type": "integer",
            "format": "int32",
            "description": "keep at most k detections",
            "minimum": 1.0,
            "maximum": 10000.0
          }
        },
        "additionalProperties": false,
        "description": "Request-level parameter overrides (strict: unknown keys -> 422)"
      },
      "Request_PPHUMAN_SEG_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "type": "object",
            "properties": {},
            "additionalProperties": false,
            "description": "This model declares no request-level parameters"
          }
        }
      },
      "Request_PP_MATTING_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "type": "object",
            "properties": {},
            "additionalProperties": false,
            "description": "This model declares no request-level parameters"
          }
        }
      },
      "Request_REAL_ESRGAN_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "type": "object",
            "properties": {},
            "additionalProperties": false,
            "description": "This model declares no request-level parameters"
          }
        }
      },
      "Request_RESNET_CLASSIFICATION_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "$ref": "#/components/schemas/Params_RESNET_CLASSIFICATION_SERVER"
          }
        }
      },
      "Params_RESNET_CLASSIFICATION_SERVER": {
        "type": "object",
        "properties": {
          "top_k": {
            "type": "integer",
            "format": "int32",
            "description": "keep the k highest scores, descending",
            "minimum": 1.0,
            "maximum": 1000.0
          }
        },
        "additionalProperties": false,
        "description": "Request-level parameter overrides (strict: unknown keys -> 422)"
      },
      "Request_SAM_AMG_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "$ref": "#/components/schemas/Params_SAM_AMG_SERVER"
          }
        }
      },
      "Params_SAM_AMG_SERVER": {
        "type": "object",
        "properties": {
          "points_per_side": {
            "type": "integer",
            "format": "int32",
            "description": "prompt point grid density (n x n points)",
            "minimum": 1.0,
            "maximum": 64.0
          },
          "pred_iou_thresh": {
            "type": "number",
            "format": "float",
            "description": "mask quality filter",
            "minimum": 0.0,
            "maximum": 1.0
          },
          "stability_score_thresh": {
            "type": "number",
            "format": "float",
            "description": "mask stability filter",
            "minimum": 0.0,
            "maximum": 1.0
          },
          "box_nms_thresh": {
            "type": "number",
            "format": "float",
            "description": "mask NMS IoU threshold",
            "minimum": 0.0,
            "maximum": 1.0
          },
          "min_mask_region_area": {
            "type": "integer",
            "format": "int32",
            "description": "drop masks smaller than this many pixels",
            "minimum": 0.0,
            "maximum": 100000.0
          }
        },
        "additionalProperties": false,
        "description": "Request-level parameter overrides (strict: unknown keys -> 422)"
      },
      "Request_SUPERPOINT_FP_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "$ref": "#/components/schemas/Params_SUPERPOINT_FP_SERVER"
          }
        }
      },
      "Params_SUPERPOINT_FP_SERVER": {
        "type": "object",
        "properties": {
          "score_threshold": {
            "type": "number",
            "format": "float",
            "description": "interest point confidence threshold",
            "minimum": 0.001,
            "maximum": 1.0
          },
          "nms_radius": {
            "type": "integer",
            "format": "int32",
            "description": "NMS suppression radius in pixels",
            "minimum": 1.0,
            "maximum": 50.0
          }
        },
        "additionalProperties": false,
        "description": "Request-level parameter overrides (strict: unknown keys -> 422)"
      },
      "Request_YOLOV5_DETECTION_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "$ref": "#/components/schemas/Params_YOLOV5_DETECTION_SERVER"
          }
        }
      },
      "Params_YOLOV5_DETECTION_SERVER": {
        "type": "object",
        "properties": {
          "score_threshold": {
            "type": "number",
            "format": "float",
            "description": "confidence threshold",
            "minimum": 0.0,
            "maximum": 1.0
          },
          "nms_threshold": {
            "type": "number",
            "format": "float",
            "description": "per-class NMS IoU threshold",
            "minimum": 0.1,
            "maximum": 1.0
          },
          "top_k": {
            "type": "integer",
            "format": "int32",
            "description": "keep at most k detections",
            "minimum": 1.0,
            "maximum": 10000.0
          }
        },
        "additionalProperties": false,
        "description": "Request-level parameter overrides (strict: unknown keys -> 422)"
      },
      "Request_YOLOV6_DETECTION_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "$ref": "#/components/schemas/Params_YOLOV6_DETECTION_SERVER"
          }
        }
      },
      "Params_YOLOV6_DETECTION_SERVER": {
        "type": "object",
        "properties": {
          "score_threshold": {
            "type": "number",
            "format": "float",
            "description": "confidence threshold",
            "minimum": 0.0,
            "maximum": 1.0
          },
          "nms_threshold": {
            "type": "number",
            "format": "float",
            "description": "per-class NMS IoU threshold",
            "minimum": 0.1,
            "maximum": 1.0
          },
          "top_k": {
            "type": "integer",
            "format": "int32",
            "description": "keep at most k detections",
            "minimum": 1.0,
            "maximum": 10000.0
          }
        },
        "additionalProperties": false,
        "description": "Request-level parameter overrides (strict: unknown keys -> 422)"
      },
      "Request_YOLOV7_DETECTION_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "$ref": "#/components/schemas/Params_YOLOV7_DETECTION_SERVER"
          }
        }
      },
      "Params_YOLOV7_DETECTION_SERVER": {
        "type": "object",
        "properties": {
          "score_threshold": {
            "type": "number",
            "format": "float",
            "description": "confidence threshold",
            "minimum": 0.0,
            "maximum": 1.0
          },
          "nms_threshold": {
            "type": "number",
            "format": "float",
            "description": "per-class NMS IoU threshold",
            "minimum": 0.1,
            "maximum": 1.0
          },
          "top_k": {
            "type": "integer",
            "format": "int32",
            "description": "keep at most k detections",
            "minimum": 1.0,
            "maximum": 10000.0
          }
        },
        "additionalProperties": false,
        "description": "Request-level parameter overrides (strict: unknown keys -> 422)"
      },
      "Request_YOLOV8_DETECTION_SERVER": {
        "type": "object",
        "required": [
          "images"
        ],
        "additionalProperties": false,
        "properties": {
          "req_id": {
            "type": "string",
            "description": "Optional trace id echoed as task_id"
          },
          "images": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "Base64 encoded image"
            },
            "minItems": 1,
            "description": "One result entry per image (index-aligned results[])"
          },
          "options": {
            "$ref": "#/components/schemas/OutputOptions"
          },
          "params": {
            "$ref": "#/components/schemas/Params_YOLOV8_DETECTION_SERVER"
          }
        }
      },
      "Params_YOLOV8_DETECTION_SERVER": {
        "type": "object",
        "properties": {
          "score_threshold": {
            "type": "number",
            "format": "float",
            "description": "confidence threshold",
            "minimum": 0.0,
            "maximum": 1.0
          },
          "nms_threshold": {
            "type": "number",
            "format": "float",
            "description": "per-class NMS IoU threshold",
            "minimum": 0.1,
            "maximum": 1.0
          },
          "top_k": {
            "type": "integer",
            "format": "int32",
            "description": "keep at most k detections",
            "minimum": 1.0,
            "maximum": 10000.0
          }
        },
        "additionalProperties": false,
        "description": "Request-level parameter overrides (strict: unknown keys -> 422)"
      },
      "UnifiedResponse": {
        "type": "object",
        "required": [
          "status",
          "status_str",
          "task_id",
          "results",
          "partial"
        ],
        "properties": {
          "status": {
            "type": "integer",
            "description": "Business status code (0 = OK)"
          },
          "status_str": {
            "type": "string"
          },
          "task_id": {
            "type": "string",
            "description": "req_id echo or server-generated id"
          },
          "model": {
            "type": "object",
            "properties": {
              "name": {
                "type": "string"
              },
              "version": {
                "type": "string"
              }
            }
          },
          "results": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/ResponseItem"
            },
            "description": "Index-aligned with the request images[]"
          },
          "server_time_ms": {
            "type": "number"
          },
          "partial": {
            "type": "boolean",
            "description": "true when the deadline hit mid-request"
          },
          "errors": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/ResponseError"
            },
            "description": "Present on 422 rejections: pointer-located violations"
          }
        }
      },
      "ResponseItem": {
        "type": "object",
        "required": [
          "status",
          "data"
        ],
        "properties": {
          "status": {
            "type": "integer",
            "description": "Per-item status; 0 = OK"
          },
          "data": {
            "nullable": true,
            "description": "Task payload; null on item failure"
          }
        }
      },
      "ResponseError": {
        "type": "object",
        "required": [
          "pointer",
          "message"
        ],
        "properties": {
          "pointer": {
            "type": "string",
            "description": "JSON pointer of the offending field"
          },
          "message": {
            "type": "string"
          }
        }
      },
      "OutputOptions": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "encoding": {
            "type": "string",
            "enum": [
              "png",
              "jpeg",
              "webp"
            ],
            "default": "png",
            "description": "Image encoding of embedded outputs"
          },
          "include_image": {
            "type": "boolean",
            "default": true
          },
          "max_results": {
            "type": "integer",
            "minimum": 0,
            "default": 0,
            "description": "0 = unlimited"
          },
          "echo_params": {
            "type": "boolean",
            "default": false
          }
        }
      },
      "EnvelopeData": {
        "nullable": true
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
            "description": "Base64 segmentation mask"
          },
          "colorized_mask": {
            "type": "string",
            "description": "Base64 colorized mask"
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
            "description": "Base64 matting result"
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
            "description": "Base64 enhanced image"
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
            "description": "Base64 colorized depth map"
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
      },
      "ImageResult": {
        "type": "object",
        "required": [
          "image"
        ],
        "properties": {
          "image": {
            "type": "string",
            "description": "Base64 generated image"
          }
        }
      },
      "SamAmgItem": {
        "type": "object",
        "required": [
          "segmentation",
          "area",
          "predicted_iou",
          "stability_score"
        ],
        "properties": {
          "segmentation": {
            "type": "string",
            "description": "Base64 mask"
          },
          "area": {
            "type": "integer"
          },
          "bbox": {
            "$ref": "#/components/schemas/BBox"
          },
          "predicted_iou": {
            "type": "number"
          },
          "stability_score": {
            "type": "number"
          }
        }
      },
      "SamAmgResult": {
        "type": "array",
        "items": {
          "$ref": "#/components/schemas/SamAmgItem"
        }
      }
    },
    "responses": {
      "BadRequest": {
        "description": "Malformed JSON body",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/UnifiedResponse"
            }
          }
        }
      },
      "Unauthorized": {
        "description": "Missing or invalid bearer token",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/UnifiedResponse"
            }
          }
        }
      },
      "NotFound": {
        "description": "Unknown path",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/UnifiedResponse"
            }
          }
        }
      },
      "MethodNotAllowed": {
        "description": "Only POST is allowed on model paths",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/UnifiedResponse"
            }
          }
        }
      },
      "PayloadTooLarge": {
        "description": "Body exceeds request_size_limit or images exceeds max_request_items",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/UnifiedResponse"
            }
          }
        }
      },
      "UnsupportedMediaType": {
        "description": "Content-Type is not application/json",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/UnifiedResponse"
            }
          }
        }
      },
      "ValidationError": {
        "description": "Strict envelope rejection: errors[] carries JSON pointers",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/UnifiedResponse"
            }
          }
        }
      },
      "RateLimited": {
        "description": "Per-item queue backpressure or per-client-IP rate limit",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/UnifiedResponse"
            }
          }
        }
      },
      "InternalError": {
        "description": "Model or server error (per-item failures keep their own results[].status)",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/UnifiedResponse"
            }
          }
        }
      },
      "NotReady": {
        "description": "Server is not ready",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/UnifiedResponse"
            }
          }
        }
      },
      "GatewayTimeout": {
        "description": "Model run timeout",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/UnifiedResponse"
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
