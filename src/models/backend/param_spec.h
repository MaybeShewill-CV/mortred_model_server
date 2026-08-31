/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: param_spec.h
 * Date: 26-8-31
 ************************************************/

#ifndef MORTRED_MODELS_BACKEND_PARAM_SPEC_H
#define MORTRED_MODELS_BACKEND_PARAM_SPEC_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace jinq {
namespace models {
namespace backend {

/*** Schema entry for one request-overridable model parameter.
 *
 * Declared by the model class (next to its preprocess/postprocess hooks),
 * aggregated by the task catalog and surfaced to OpenAPI. The config TOML
 * stays the default source of values; a request may only override keys that
 * are declared here and request_overridable.
 *
 * Authoring style mirrors SessionSpec / IoSpec: a small fluent builder so
 * the whole surface of one model reads as a table.
 */
struct ParamSpec {
    enum class Type { F32, I32, BOOL, STRING };

    std::string key;
    Type type = Type::F32;
    std::string description;
    bool request_overridable = true;

    // numeric range constraint (F32 / I32), inclusive bounds
    bool has_range = false;
    double range_min = 0.0;
    double range_max = 0.0;

    // STRING enum constraint; empty = any non-empty string is accepted
    std::vector<std::string> enum_values;

    static ParamSpec f32(std::string key) { return ParamSpec{std::move(key), Type::F32}; }
    static ParamSpec i32(std::string key) { return ParamSpec{std::move(key), Type::I32}; }
    static ParamSpec boolean(std::string key) { return ParamSpec{std::move(key), Type::BOOL}; }
    static ParamSpec str(std::string key) { return ParamSpec{std::move(key), Type::STRING}; }

    ParamSpec &&range(double min_value, double max_value) && {
        has_range = true;
        range_min = min_value;
        range_max = max_value;
        return std::move(*this);
    }

    ParamSpec &&values(std::vector<std::string> allowed) && {
        enum_values = std::move(allowed);
        return std::move(*this);
    }

    ParamSpec &&desc(std::string text) && {
        description = std::move(text);
        return std::move(*this);
    }

    ParamSpec &&config_only() && {
        request_overridable = false;
        return std::move(*this);
    }

    const char *type_name() const {
        switch (type) {
            case Type::F32:
                return "f32";
            case Type::I32:
                return "i32";
            case Type::BOOL:
                return "bool";
            case Type::STRING:
                return "string";
        }
        return "unknown";
    }

  private:
    explicit ParamSpec(std::string k, Type t) : key(std::move(k)), type(t) {}
};

/*** JSON-typed parameter candidate produced by the request parser. The kind
 * mirrors what the envelope actually carried, so strict type checking can
 * distinguish "1" (text) from 1 (integer) and true from 1. */
struct ParamValue {
    enum class Kind { F64, I64, BOOL, TEXT };

    Kind kind = Kind::TEXT;
    double f64 = 0.0;
    int64_t i64 = 0;
    bool boolean = false;
    std::string text;

    static ParamValue of(double value) {
        ParamValue v;
        v.kind = Kind::F64;
        v.f64 = value;
        return v;
    }

    static ParamValue of(int64_t value) {
        ParamValue v;
        v.kind = Kind::I64;
        v.i64 = value;
        return v;
    }

    static ParamValue of(bool value) {
        ParamValue v;
        v.kind = Kind::BOOL;
        v.boolean = value;
        return v;
    }

    static ParamValue of(std::string value) {
        ParamValue v;
        v.kind = Kind::TEXT;
        v.text = std::move(value);
        return v;
    }
};

/*** One strict-validation failure: a JSON pointer relative to the request's
 * params object plus a human readable constraint message. */
struct ParamViolation {
    std::string pointer;
    std::string message;
};

/*** Validated, request-scoped parameter set.
 *
 * Storage is a flat small vector with linear lookup: request parameters are
 * few (<= k_max_params) and every inference call reads them, so per-request
 * node allocations (std::map / unordered_map) are a measurable cost at high
 * QPS. Typed getters take the config-derived fallback so a model reads both
 * sources in one expression:
 *
 *   float thr = context.params->get_f32("score_threshold", _m_cfg_score_thr);
 */
class ParamSet {
  public:
    static constexpr size_t k_max_params = 16;

    bool set_f32(const std::string &key, float value) { return insert(key, ParamValue::of(static_cast<double>(value))); }
    bool set_i32(const std::string &key, int32_t value) { return insert(key, ParamValue::of(static_cast<int64_t>(value))); }
    bool set_bool(const std::string &key, bool value) { return insert(key, ParamValue::of(value)); }
    bool set_str(const std::string &key, std::string value) { return insert(key, ParamValue::of(std::move(value))); }

    float get_f32(const std::string &key, float fallback) const {
        const Entry *entry = find(key);
        if (entry == nullptr || entry->value.kind != ParamValue::Kind::F64) {
            return fallback;
        }
        return static_cast<float>(entry->value.f64);
    }

    int32_t get_i32(const std::string &key, int32_t fallback) const {
        const Entry *entry = find(key);
        if (entry == nullptr || entry->value.kind != ParamValue::Kind::I64) {
            return fallback;
        }
        return static_cast<int32_t>(entry->value.i64);
    }

    bool get_bool(const std::string &key, bool fallback) const {
        const Entry *entry = find(key);
        if (entry == nullptr || entry->value.kind != ParamValue::Kind::BOOL) {
            return fallback;
        }
        return entry->value.boolean;
    }

    std::string get_str(const std::string &key, const std::string &fallback) const {
        const Entry *entry = find(key);
        if (entry == nullptr || entry->value.kind != ParamValue::Kind::TEXT) {
            return fallback;
        }
        return entry->value.text;
    }

    bool contains(const std::string &key) const { return find(key) != nullptr; }

    bool empty() const { return _m_entries.empty(); }

    size_t size() const { return _m_entries.size(); }

    std::vector<std::string> keys() const {
        std::vector<std::string> out;
        out.reserve(_m_entries.size());
        for (const auto &entry : _m_entries) {
            out.push_back(entry.key);
        }
        return out;
    }

  private:
    struct Entry {
        std::string key;
        ParamValue value;
    };

    const Entry *find(const std::string &key) const {
        for (const auto &entry : _m_entries) {
            if (entry.key == key) {
                return &entry;
            }
        }
        return nullptr;
    }

    Entry *find(const std::string &key) {
        for (auto &entry : _m_entries) {
            if (entry.key == key) {
                return &entry;
            }
        }
        return nullptr;
    }

    /*** first write wins: a duplicate key is a request bug and is rejected
     * by validate_params() before the set is ever built, so insert() only
     * guards the capacity here. */
    bool insert(const std::string &key, ParamValue value) {
        if (key.empty() || _m_entries.size() >= k_max_params || find(key) != nullptr) {
            return false;
        }
        _m_entries.push_back(Entry{key, std::move(value)});
        return true;
    }

    std::vector<Entry> _m_entries;
};

namespace detail {

inline std::string join_keys(const std::vector<ParamSpec> &specs) {
    std::string out;
    for (size_t idx = 0; idx < specs.size(); ++idx) {
        if (idx != 0) {
            out += ", ";
        }
        out += specs[idx].key;
    }
    return out;
}

inline std::string join_values(const std::vector<std::string> &values) {
    std::string out;
    for (size_t idx = 0; idx < values.size(); ++idx) {
        if (idx != 0) {
            out += ", ";
        }
        out += values[idx];
    }
    return out;
}

inline std::string number_to_string(double value) {
    // shortest round-trip form is enough for diagnostics
    return std::to_string(value);
}

} // namespace detail

/*** Strict validation of request-provided values against the declared
 * schema. Unknown keys, config-only keys, type mismatches, range/enum
 * violations and duplicates are rejected - a misspelled parameter name must
 * fail loudly instead of silently doing nothing. On success `out` receives
 * the canonical set; on any violation `out` is left untouched. Returns an
 * empty vector iff every candidate was accepted.
 */
inline std::vector<ParamViolation> validate_params(const std::vector<ParamSpec> &specs,
                                                   const std::vector<std::pair<std::string, ParamValue>> &candidates,
                                                   ParamSet *out) {
    std::vector<ParamViolation> violations;

    if (candidates.size() > ParamSet::k_max_params) {
        violations.push_back({"/", "too many parameters (max " + std::to_string(ParamSet::k_max_params) + ")"});
        return violations;
    }

    std::vector<std::string> seen;

    for (const auto &candidate : candidates) {
        const std::string &key = candidate.first;
        const ParamValue &value = candidate.second;
        const std::string pointer = "/" + key;

        for (const auto &duplicate : seen) {
            if (duplicate == key) {
                violations.push_back({pointer, "duplicate parameter '" + key + "'"});
                break;
            }
        }
        if (!violations.empty() && violations.back().pointer == pointer &&
            violations.back().message.rfind("duplicate parameter", 0) == 0) {
            continue;
        }
        seen.push_back(key);

        const ParamSpec *spec = nullptr;
        for (const auto &entry : specs) {
            if (entry.key == key) {
                spec = &entry;
                break;
            }
        }
        if (spec == nullptr) {
            violations.push_back({pointer, "unknown parameter '" + key + "'; allowed: " + detail::join_keys(specs)});
            continue;
        }
        if (!spec->request_overridable) {
            violations.push_back({pointer, "parameter '" + key + "' is configuration-only and cannot be set per request"});
            continue;
        }

        const bool is_number = value.kind == ParamValue::Kind::F64 || value.kind == ParamValue::Kind::I64;
        const double number = value.kind == ParamValue::Kind::I64 ? static_cast<double>(value.i64) : value.f64;

        switch (spec->type) {
            case ParamSpec::Type::F32:
                if (!is_number) {
                    violations.push_back({pointer, "parameter '" + key + "' must be a number"});
                    continue;
                }
                if (spec->has_range && (number < spec->range_min || number > spec->range_max)) {
                    violations.push_back({pointer, "parameter '" + key + "' must be in [" +
                                                          detail::number_to_string(spec->range_min) + ", " +
                                                          detail::number_to_string(spec->range_max) + "]"});
                    continue;
                }
                break;

            case ParamSpec::Type::I32:
                if (value.kind != ParamValue::Kind::I64) {
                    violations.push_back(
                        {pointer, "parameter '" + key + "' must be an integer (no decimal part, no bool/string)"});
                    continue;
                }
                if (spec->has_range && (number < spec->range_min || number > spec->range_max)) {
                    violations.push_back({pointer, "parameter '" + key + "' must be in [" +
                                                          detail::number_to_string(spec->range_min) + ", " +
                                                          detail::number_to_string(spec->range_max) + "]"});
                    continue;
                }
                break;

            case ParamSpec::Type::BOOL:
                if (value.kind != ParamValue::Kind::BOOL) {
                    violations.push_back({pointer, "parameter '" + key + "' must be a boolean"});
                    continue;
                }
                break;

            case ParamSpec::Type::STRING:
                if (value.kind != ParamValue::Kind::TEXT) {
                    violations.push_back({pointer, "parameter '" + key + "' must be a string"});
                    continue;
                }
                if (value.text.empty()) {
                    violations.push_back({pointer, "parameter '" + key + "' must be a non-empty string"});
                    continue;
                }
                if (!spec->enum_values.empty()) {
                    bool matched = false;
                    for (const auto &allowed : spec->enum_values) {
                        if (allowed == value.text) {
                            matched = true;
                            break;
                        }
                    }
                    if (!matched) {
                        violations.push_back(
                            {pointer, "parameter '" + key + "' must be one of: " + detail::join_values(spec->enum_values)});
                        continue;
                    }
                }
                break;
        }
    }

    if (!violations.empty()) {
        return violations;
    }

    if (out != nullptr) {
        for (const auto &candidate : candidates) {
            const std::string &key = candidate.first;
            const ParamValue &value = candidate.second;
            const ParamSpec *spec = nullptr;
            for (const auto &entry : specs) {
                if (entry.key == key) {
                    spec = &entry;
                    break;
                }
            }
            if (spec == nullptr) {
                continue;
            }
            // storage is canonicalised by the DECLARED type, not by the
            // candidate kind: an integer literal accepted by an f32 spec
            // must be readable through get_f32()
            switch (spec->type) {
                case ParamSpec::Type::F32:
                    out->set_f32(key, static_cast<float>(value.kind == ParamValue::Kind::I64
                                                                 ? static_cast<double>(value.i64)
                                                                 : value.f64));
                    break;
                case ParamSpec::Type::I32:
                    out->set_i32(key, static_cast<int32_t>(value.i64));
                    break;
                case ParamSpec::Type::BOOL:
                    out->set_bool(key, value.boolean);
                    break;
                case ParamSpec::Type::STRING:
                    out->set_str(key, value.text);
                    break;
            }
        }
    }
    return violations;
}

} // namespace backend
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_BACKEND_PARAM_SPEC_H
