/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: time_stamp.h
* Date: 22-6-5
************************************************/

#ifndef MORTRED_MODEL_SERVER_TIMESTAMP_H
#define MORTRED_MODEL_SERVER_TIMESTAMP_H

#include <chrono>
#include <cstdint>
#include <string>

namespace jinq {
namespace common {

// thin wrapper over system_clock time_point with microsecond precision
class Timestamp {
public:
    using clock = std::chrono::system_clock;
    using time_point = std::chrono::time_point<clock, std::chrono::microseconds>;

    Timestamp();
    Timestamp(const Timestamp& that) = default;
    Timestamp& operator=(const Timestamp& that) = default;

    /***
     * @param micro_sec_since_epoch microseconds from 1970-01-01 00:00:00
     */
    explicit Timestamp(uint64_t micro_sec_since_epoch);

    std::string to_str() const;
    std::string to_format_str() const;
    std::string to_format_str(const char* fmt) const;
    uint64_t micro_sec_since_epoch() const;

    bool valid() const {
        return micro_sec_since_epoch() > 0;
    }

    static Timestamp now();
    static Timestamp invalid() {
        return Timestamp();
    }

    static const int k_micro_sec_per_sec = 1000 * 1000;

private:
    explicit Timestamp(time_point tp);

    time_point _m_time_point;
};

inline bool operator<(const Timestamp& lhs, const Timestamp& rhs) {
    return lhs.micro_sec_since_epoch() < rhs.micro_sec_since_epoch();
}

inline bool operator>(const Timestamp& lhs, const Timestamp& rhs) {
    return lhs.micro_sec_since_epoch() > rhs.micro_sec_since_epoch();
}

inline bool operator<=(const Timestamp& lhs, const Timestamp& rhs) {
    return lhs.micro_sec_since_epoch() <= rhs.micro_sec_since_epoch();
}

inline bool operator>=(const Timestamp& lhs, const Timestamp& rhs) {
    return lhs.micro_sec_since_epoch() >= rhs.micro_sec_since_epoch();
}

inline bool operator==(const Timestamp& lhs, const Timestamp& rhs) {
    return lhs.micro_sec_since_epoch() == rhs.micro_sec_since_epoch();
}

inline bool operator!=(const Timestamp& lhs, const Timestamp& rhs) {
    return lhs.micro_sec_since_epoch() != rhs.micro_sec_since_epoch();
}

// elapsed seconds between two timestamps
inline double operator-(const Timestamp& high, const Timestamp& low) {
    uint64_t diff = high.micro_sec_since_epoch() - low.micro_sec_since_epoch();
    return static_cast<double>(diff) / Timestamp::k_micro_sec_per_sec;
}

}  // namespace common
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_TIMESTAMP_H
