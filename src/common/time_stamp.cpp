/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: time_stamp.cpp
* Date: 22-6-5
************************************************/

#include "time_stamp.h"

namespace jinq {
namespace common {
static_assert(sizeof(Timestamp) == sizeof(uint64_t), "Timestamp should be same size as uint64_t");

Timestamp::Timestamp()
        : _m_micro_sec_since_epoch(0) {
}

Timestamp::Timestamp(uint64_t micro_sec_since_epoch)
        : _m_micro_sec_since_epoch(micro_sec_since_epoch) {
}

Timestamp::Timestamp(const Timestamp &that)
        : _m_micro_sec_since_epoch(that._m_micro_sec_since_epoch) {
}

Timestamp &Timestamp::operator=(const Timestamp &that) {
    _m_micro_sec_since_epoch = that._m_micro_sec_since_epoch;
    return *this;
}

void Timestamp::swap(Timestamp &that) {
    std::swap(_m_micro_sec_since_epoch, that._m_micro_sec_since_epoch);
}

std::string Timestamp::to_str() const {
    return std::to_string(_m_micro_sec_since_epoch / k_micro_sec_per_sec)
           + "." + std::to_string(_m_micro_sec_since_epoch % k_micro_sec_per_sec);
}

std::string Timestamp::to_format_str() const {
    return to_format_str("%Y-%m-%d %X");
}

std::string Timestamp::to_format_str(const char *fmt) const {
    std::time_t time = _m_micro_sec_since_epoch / k_micro_sec_per_sec;  // ms --> s
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), fmt);
    return ss.str();
}

uint64_t Timestamp::micro_sec_since_epoch() const {
    return _m_micro_sec_since_epoch;
}

Timestamp Timestamp::now() {
    uint64_t timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    return Timestamp(timestamp);
}
}
}