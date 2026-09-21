/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: base64.h
* Date: 22-6-2
************************************************/

#ifndef MORTRED_MODEL_SERVER_BASE64_H
#define MORTRED_MODEL_SERVER_BASE64_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "glog/logging.h"

namespace jinq {
namespace common {
namespace base64 {

namespace detail {

// O(1) decode lookup table; 0xFF marks invalid characters
inline const std::array<unsigned char, 256>& decode_table() {
    static const std::array<unsigned char, 256> k_table = []() {
        std::array<unsigned char, 256> table{};
        table.fill(0xFF);
        const char* alphabet =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (unsigned char i = 0; i < 64; ++i) {
            table[static_cast<unsigned char>(alphabet[i])] = i;
        }
        return table;
    }();
    return k_table;
}

}  // namespace detail

// encode raw bytes into RFC 4648 base64 (padded)
inline std::string encode(const void* data, size_t len) {
    static const char k_alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    if (data == nullptr || len == 0) {
        return std::string();
    }
    const auto* bytes = static_cast<const unsigned char*>(data);
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        const bool has_second = i + 1 < len;
        const bool has_third = i + 2 < len;
        uint32_t block = bytes[i] << 16;
        if (has_second) {
            block |= bytes[i + 1] << 8;
        }
        if (has_third) {
            block |= bytes[i + 2];
        }
        out.push_back(k_alphabet[(block >> 18) & 0x3F]);
        out.push_back(k_alphabet[(block >> 12) & 0x3F]);
        out.push_back(has_second ? k_alphabet[(block >> 6) & 0x3F] : '=');
        out.push_back(has_third ? k_alphabet[block & 0x3F] : '=');
    }
    return out;
}

inline std::string encode(std::string_view input) {
    return encode(input.data(), input.size());
}

// decode base64 into raw bytes; returns an empty string on invalid input
inline std::string decode(std::string_view input) {
    const auto& table = detail::decode_table();
    if (input.empty()) {
        return std::string();
    }

    size_t pad = 0;
    if (input.back() == '=') {
        ++pad;
        if (input.size() >= 2 && input[input.size() - 2] == '=') {
            ++pad;
        }
    }
    const size_t data_len = input.size() - pad;
    const size_t expect_rem = pad == 0 ? 0 : (pad == 1 ? 3 : 2);
    if (data_len % 4 != expect_rem) {
        LOG(ERROR) << "invalid base64 input length: " << input.size();
        return std::string();
    }

    // block-wise decode into a preallocated buffer: ~2x the per-char
    // push_back loop on multi-hundred-KB bodies (the handler-thread hot
    // path for json-envelope requests)
    std::string out;
    const size_t out_len = (data_len / 4) * 3 + (pad == 0 ? 0 : 3 - pad);
    out.resize(out_len);
    size_t w = 0;
    size_t i = 0;
    for (; i + 4 <= data_len; i += 4) {
        const uint32_t v0 = table[static_cast<unsigned char>(input[i])];
        const uint32_t v1 = table[static_cast<unsigned char>(input[i + 1])];
        const uint32_t v2 = table[static_cast<unsigned char>(input[i + 2])];
        const uint32_t v3 = table[static_cast<unsigned char>(input[i + 3])];
        if ((v0 | v1 | v2 | v3) > 63u) {  // valid entries are 0..63; 0xFF marks invalid
            LOG(ERROR) << "invalid base64 character at offset " << i;
            return std::string();
        }
        const uint32_t block = (v0 << 18) | (v1 << 12) | (v2 << 6) | v3;
        out[w++] = static_cast<char>((block >> 16) & 0xFF);
        out[w++] = static_cast<char>((block >> 8) & 0xFF);
        out[w++] = static_cast<char>(block & 0xFF);
    }
    uint32_t tail = 0;
    unsigned int shift = 0;
    for (; i < data_len; ++i) {
        const uint32_t value = table[static_cast<unsigned char>(input[i])];
        if (value > 63u) {
            LOG(ERROR) << "invalid base64 character at offset " << i;
            return std::string();
        }
        tail = (tail << 6) | value;
        shift += 6;
    }
    if (shift == 12) {
        out[w++] = static_cast<char>((tail >> 4) & 0xFF);
    } else if (shift == 18) {
        out[w++] = static_cast<char>((tail >> 10) & 0xFF);
        out[w++] = static_cast<char>((tail >> 2) & 0xFF);
    }
    return out;
}

}  // namespace base64
}  // namespace common
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_BASE64_H
