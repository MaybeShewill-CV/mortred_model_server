/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: time_stamp.cpp
 * Date: 22-6-5
 ************************************************/

#include "time_stamp.h"

#include <ctime>

namespace jinq {
namespace common {

Timestamp::Timestamp()
    : _m_time_point(std::chrono::microseconds(0)) {
}

Timestamp::Timestamp(uint64_t micro_sec_since_epoch)
    : _m_time_point(std::chrono::microseconds(micro_sec_since_epoch)) {
}

Timestamp::Timestamp(time_point tp)
    : _m_time_point(tp) {
}

uint64_t Timestamp::micro_sec_since_epoch() const {
    return static_cast<uint64_t>(_m_time_point.time_since_epoch().count());
}

Timestamp Timestamp::now() {
    return Timestamp(
        std::chrono::time_point_cast<std::chrono::microseconds>(clock::now()));
}

std::string Timestamp::to_str() const {
    return std::to_string(_m_time_point.time_since_epoch().count() / k_micro_sec_per_sec)
           + "." + std::to_string(_m_time_point.time_since_epoch().count() % k_micro_sec_per_sec);
}

std::string Timestamp::to_format_str() const {
    return to_format_str("%Y-%m-%d %X");
}

std::string Timestamp::to_format_str(const char* fmt) const {
    std::time_t seconds = std::chrono::duration_cast<std::chrono::seconds>(
                              _m_time_point.time_since_epoch())
                              .count();
    std::tm tm_buf{};
    // localtime_r: thread-safe, no static buffer (Linux only, per project scope)
    ::localtime_r(&seconds, &tm_buf);
    char buf[128];
    std::strftime(buf, sizeof(buf), fmt, &tm_buf);
    return std::string(buf);
}

}  // namespace common
}  // namespace jinq
