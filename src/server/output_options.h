/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: output_options.h
 * Date: 26-8-31
 ************************************************/

#ifndef MORTRED_SERVER_OUTPUT_OPTIONS_H
#define MORTRED_SERVER_OUTPUT_OPTIONS_H

#include <string>
#include <vector>

#include "rapidjson/document.h"

#include "models/backend/param_spec.h"

namespace jinq {
namespace server {

using ParamViolation = jinq::models::backend::ParamViolation;

/*** Request-level output options of the unified envelope.
 *
 * The only live key is encoding (used by image-bearing serializers). Unknown
 * keys 422. The struct is plain data so it can travel inside InferenceTask
 * by value.
 */
struct OutputOptions {
    enum class ImageEncoding { PNG, JPEG, WEBP };

    ImageEncoding encoding = ImageEncoding::PNG;

    const char *encoding_name() const {
        switch (encoding) {
            case ImageEncoding::PNG:
                return "png";
            case ImageEncoding::JPEG:
                return "jpeg";
            case ImageEncoding::WEBP:
                return "webp";
        }
        return "png";
    }

    /*** cv::imencode extension of the selected encoding */
    const char *encoding_extension() const {
        switch (encoding) {
            case ImageEncoding::PNG:
                return ".png";
            case ImageEncoding::JPEG:
                return ".jpg";
            case ImageEncoding::WEBP:
                return ".webp";
        }
        return ".png";
    }
};

namespace detail {

inline const char *k_allowed_option_keys = "encoding";

} // namespace detail

/*** Strict parse of the request "options" object.
 *
 * Violation pointers are relative to the options object itself ("/encoding");
 * the envelope parser prefixes them with "/options". Returns an empty vector
 * iff every member was accepted; `out` is only filled on success.
 */
inline std::vector<ParamViolation> parse_output_options(const rapidjson::Value &options, OutputOptions *out) {
    std::vector<ParamViolation> violations;
    OutputOptions parsed;

    for (auto member = options.MemberBegin(); member != options.MemberEnd(); ++member) {
        const std::string key(member->name.GetString(), member->name.GetStringLength());
        const std::string pointer = "/" + key;
        const rapidjson::Value &value = member->value;

        if (key == "encoding") {
            if (!value.IsString()) {
                violations.push_back({pointer, "option 'encoding' must be a string"});
                continue;
            }
            const std::string text(value.GetString(), value.GetStringLength());
            if (text == "png") {
                parsed.encoding = OutputOptions::ImageEncoding::PNG;
            } else if (text == "jpeg") {
                parsed.encoding = OutputOptions::ImageEncoding::JPEG;
            } else if (text == "webp") {
                parsed.encoding = OutputOptions::ImageEncoding::WEBP;
            } else {
                violations.push_back({pointer, "option 'encoding' must be one of: png, jpeg, webp"});
            }
        } else {
            violations.push_back(
                {pointer, "unknown option '" + key + "'; allowed: " + detail::k_allowed_option_keys});
        }
    }

    if (violations.empty() && out != nullptr) {
        *out = parsed;
    }
    return violations;
}

} // namespace server
} // namespace jinq

#endif // MORTRED_SERVER_OUTPUT_OPTIONS_H
