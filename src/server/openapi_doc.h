/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: openapi_doc.h
* Date: 26-8-31
************************************************/

// The full OpenAPI document is not embedded in the binary/header.
// docs/openapi.json remains the single generated artifact (scripts/gen_openapi.py).
// This header loads that file once at first use (path from MORTRED_OPENAPI_JSON
// or relative docs/openapi.json from the process CWD / known install layouts).

#ifndef MORTRED_SERVER_OPENAPI_DOC_H
#define MORTRED_SERVER_OPENAPI_DOC_H

#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>

namespace mortred {
namespace server {
namespace openapi_doc_detail {

inline std::string load_openapi_json() {
    const char* env = std::getenv("MORTRED_OPENAPI_JSON");
    const char* candidates[] = {
        env && env[0] ? env : nullptr,
        "docs/openapi.json",
        "/etc/mortred/openapi.json",
        nullptr,
    };
    for (const char* path : candidates) {
        if (path == nullptr) {
            continue;
        }
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            continue;
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        const std::string body = ss.str();
        if (!body.empty()) {
            return body;
        }
    }
    return "{\"openapi\":\"3.0.3\",\"info\":{\"title\":\"Mortred\",\"version\":\"unknown\"},"
           "\"paths\":{},\"x-mortred-error\":\"openapi.json not found; set MORTRED_OPENAPI_JSON\"}";
}

}  // namespace openapi_doc_detail

inline const std::string& k_openapi_doc_json_ref() {
    static std::once_flag once;
    static std::string doc;
    std::call_once(once, [] { doc = openapi_doc_detail::load_openapi_json(); });
    return doc;
}

// Compatibility alias used by base_server_impl.h (was a compile-time string).
inline const std::string& k_openapi_doc_json = k_openapi_doc_json_ref();

}  // namespace server
}  // namespace mortred

#endif  // MORTRED_SERVER_OPENAPI_DOC_H
