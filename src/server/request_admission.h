/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: request_admission.h
* Date: 26-9-9
************************************************/

// Orthogonal admission predicates for the model HTTP surface. Pure functions:
// no WFHttpTask, no metrics, no response assembly. 422 belongs to
// parsed_request.h; 401 / QPS 429 stay on the orchestrator front door.

#ifndef MORTRED_SERVER_REQUEST_ADMISSION_H
#define MORTRED_SERVER_REQUEST_ADMISSION_H

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace jinq {
namespace server {

enum class ContentTypeKind { Json, RawBody, Unsupported };

inline std::string normalized_media_type(std::string content_type) {
    std::transform(content_type.begin(), content_type.end(), content_type.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const auto semi = content_type.find(';');
    if (semi != std::string::npos) {
        content_type = content_type.substr(0, semi);
    }
    size_t b = 0;
    size_t e = content_type.size();
    while (b < e && std::isspace(static_cast<unsigned char>(content_type[b]))) {
        ++b;
    }
    while (e > b && std::isspace(static_cast<unsigned char>(content_type[e - 1]))) {
        --e;
    }
    return content_type.substr(b, e - b);
}

inline ContentTypeKind content_type_kind(const std::string& content_type) {
    const std::string ct = normalized_media_type(content_type);
    if (ct == "application/json") {
        return ContentTypeKind::Json;
    }
    if (ct.rfind("image/", 0) == 0 || ct == "application/octet-stream") {
        return ContentTypeKind::RawBody;
    }
    return ContentTypeKind::Unsupported;
}

inline bool declared_body_exceeds(const std::string& content_length_str, size_t limit_mb) {
    if (content_length_str.empty()) {
        return false;
    }
    char* end = nullptr;
    const unsigned long long declared =
        std::strtoull(content_length_str.c_str(), &end, 10);
    if (end == content_length_str.c_str() || *end != '\0') {
        return false;
    }
    return declared > limit_mb * 1024ULL * 1024ULL;
}

inline bool item_count_exceeds(size_t n_items, size_t max_request_items) {
    return n_items > max_request_items;
}

inline bool queue_would_overflow(size_t waiting_jobs, size_t n_items, int max_queue_depth) {
    if (max_queue_depth <= 0) {
        return false;
    }
    return waiting_jobs + n_items > static_cast<size_t>(max_queue_depth);
}

}  // namespace server
}  // namespace jinq

#endif  // MORTRED_SERVER_REQUEST_ADMISSION_H
