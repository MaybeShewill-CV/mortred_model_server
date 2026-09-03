/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: management_envelope.h
 * Date: 26-9-4
 ************************************************/

// Management-plane rewrite of the unified data-plane envelope. Supervisor
// /api/v1/infer, /api/v1/jobs and /api/v1/pipelines must speak
// {req_id, images[], params, options} / results[].data — never the removed
// img_data field and never the legacy {data: ...} response shape.
//
// Header-only and workflow-free so the rewrite is unit-testable in every CI
// configuration. The img_data migration text is byte-identical to
// jinq::server::detail::k_img_data_migration (locked by pipeline_contract_unittest).

#ifndef MORTRED_CONTROL_MANAGEMENT_ENVELOPE_H
#define MORTRED_CONTROL_MANAGEMENT_ENVELOPE_H

#include <string>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace mortred {
namespace control {

inline const char *k_img_data_migration =
    "field 'img_data' was removed; use images: [\"<base64>\"] (migration: img_data -> images[0])";

struct EnvelopeRewrite {
    bool ok = false;
    int http_status = 400;
    std::string pointer;
    std::string message;
    std::string json;
};

inline std::string dump_json(const rapidjson::Value &value) {
    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
    value.Accept(writer);
    return buf.GetString();
}

/*** Copy only the unified request-envelope keys. Management-only fields
 * (server_id, model, steps, ...) are dropped. img_data is a hard 422 even
 * when images[] is also present — same fail-fast rule as the data plane. */
inline EnvelopeRewrite copy_request_envelope(const rapidjson::Value &doc) {
    EnvelopeRewrite out;
    if (!doc.IsObject()) {
        out.pointer = "/";
        out.message = "request body must be a JSON object";
        return out;
    }
    if (doc.HasMember("img_data")) {
        out.http_status = 422;
        out.pointer = "/img_data";
        out.message = k_img_data_migration;
        return out;
    }

    rapidjson::Document next;
    next.SetObject();
    auto &allocator = next.GetAllocator();
    static const char *k_keys[] = {"req_id", "images", "params", "options"};
    for (const char *key : k_keys) {
        if (doc.HasMember(key)) {
            next.AddMember(rapidjson::Value(key, allocator), rapidjson::Value(doc[key], allocator),
                           allocator);
        }
    }
    if (!next.HasMember("images")) {
        out.pointer = "/images";
        out.message = "required field 'images' is missing";
        return out;
    }

    out.ok = true;
    out.http_status = 200;
    out.json = dump_json(next);
    return out;
}

/*** Build {images: [...]} from results[0].data[field] of a unified response.
 * The legacy {data: {field}} shape is refused so pipelines cannot silently
 * target the old envelope. */
inline EnvelopeRewrite extract_prev_output_images(const std::string &prev_json,
                                                  const std::string &field) {
    EnvelopeRewrite out;
    const std::string missing = "cannot extract '" + field +
                                "' from previous output; expected unified results[].data";
    rapidjson::Document prev;
    prev.Parse(prev_json.c_str());
    if (prev.HasParseError() || !prev.IsObject()) {
        out.message = missing;
        return out;
    }
    if (!prev.HasMember("results") || !prev["results"].IsArray() || prev["results"].Empty()) {
        out.message = missing;
        return out;
    }

    const rapidjson::Value &item = prev["results"][0];
    if (!item.IsObject() || !item.HasMember("data") || item["data"].IsNull()) {
        out.message = "cannot extract '" + field +
                      "' from previous output; results[0].data is missing";
        return out;
    }
    const rapidjson::Value &data = item["data"];
    if (!data.IsObject() || !data.HasMember(field.c_str())) {
        out.message = "cannot extract '" + field + "' from previous output";
        return out;
    }

    const rapidjson::Value &value = data[field.c_str()];
    rapidjson::Document next;
    next.SetObject();
    auto &allocator = next.GetAllocator();
    rapidjson::Value images(rapidjson::kArrayType);

    if (value.IsString() && value.GetStringLength() > 0) {
        images.PushBack(rapidjson::Value(value, allocator), allocator);
    } else if (value.IsArray() && !value.Empty()) {
        for (const auto &element : value.GetArray()) {
            if (!element.IsString() || element.GetStringLength() == 0) {
                out.message = "cannot extract '" + field +
                              "': expected a base64 string or array of strings";
                return out;
            }
            images.PushBack(rapidjson::Value(element, allocator), allocator);
        }
    } else {
        out.message =
            "cannot extract '" + field + "': expected a base64 string or array of strings";
        return out;
    }

    next.AddMember("images", images, allocator);
    out.ok = true;
    out.http_status = 200;
    out.json = dump_json(next);
    return out;
}

/*** Pipeline step input: "prev_output.<field>" extracts from the previous
 * unified response; any other key leaves the current body unchanged (the
 * first step is already a request envelope). */
inline EnvelopeRewrite apply_pipeline_step_input(const std::string &current_body,
                                                 const std::string &input_key) {
    if (input_key.rfind("prev_output.", 0) == 0) {
        return extract_prev_output_images(current_body, input_key.substr(12));
    }
    EnvelopeRewrite out;
    out.ok = true;
    out.http_status = 200;
    out.json = current_body;
    return out;
}

} // namespace control
} // namespace mortred

#endif // MORTRED_CONTROL_MANAGEMENT_ENVELOPE_H
