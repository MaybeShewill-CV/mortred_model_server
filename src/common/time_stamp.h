/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: time_stamp.h
* Date: 22-6-5
************************************************/

#ifndef MMAISERVER_TIMESTAMP_H
#define MMAISERVER_TIMESTAMP_H

#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>

namespace mortred {
namespace common {
class Timestamp {
public:
    /***
     *
     */
    Timestamp();

    /***
     *
     * @param that
     */
    Timestamp(const Timestamp &that);

    /***
     *
     * @param that
     * @return
     */
    Timestamp &operator=(const Timestamp &that);

    /***
     *
     * @param micro_sec_since_epoch: The microseconds from 1970-01-01 00:00:00.
     */
    explicit Timestamp(uint64_t micro_sec_since_epoch);

    /***
     *
     * @param that
     */
    void swap(Timestamp &that);

    /***
     *
     * @return
     */
    std::string to_str() const;

    /***
     *
     * @return
     */
    std::string to_format_str() const;

    /***
     *
     * @param fmt
     * @return
     */
    std::string to_format_str(const char *fmt) const;

    /***
     *
     * @return
     */
    uint64_t micro_sec_since_epoch() const;

    /***
     *
     * @return
     */
    inline bool valid() const {
        return _m_micro_sec_since_epoch > 0;
    }

    /***
     *
     * @return
     */
    static Timestamp now();

    /***
     *
     * @return
     */
    static Timestamp invalid() {
        return {};
    }

    static const int k_micro_sec_per_sec = 1000 * 1000;
private:
    uint64_t _m_micro_sec_since_epoch;
};

/***
 *
 * @param lhs
 * @param rhs
 * @return
 */
inline bool operator<(const Timestamp& lhs, const Timestamp& rhs) {
    return lhs.micro_sec_since_epoch() < rhs.micro_sec_since_epoch();
}

/***
 *
 * @param lhs
 * @param rhs
 * @return
 */
inline bool operator>(const Timestamp& lhs, const Timestamp& rhs) {
    return lhs.micro_sec_since_epoch() > rhs.micro_sec_since_epoch();
}

/***
 *
 * @param lhs
 * @param rhs
 * @return
 */
inline bool operator<=(const Timestamp& lhs, const Timestamp& rhs) {
    return lhs.micro_sec_since_epoch() <= rhs.micro_sec_since_epoch();
}

/***
 *
 * @param lhs
 * @param rhs
 * @return
 */
inline bool operator>=(const Timestamp& lhs, const Timestamp& rhs) {
    return lhs.micro_sec_since_epoch() >= rhs.micro_sec_since_epoch();
}

/***
 *
 * @param lhs
 * @param rhs
 * @return
 */
inline bool operator==(const Timestamp& lhs, const Timestamp& rhs) {
    return lhs.micro_sec_since_epoch() == rhs.micro_sec_since_epoch();
}

/***
 *
 * @param lhs
 * @param rhs
 * @return
 */
inline bool operator!=(const Timestamp& lhs, const Timestamp& rhs) {
    return lhs.micro_sec_since_epoch() != rhs.micro_sec_since_epoch();
}

/***
 *
 * @param lhs
 * @param ms
 * @return
 */
inline Timestamp operator+(const Timestamp& lhs, uint64_t ms) {
    return Timestamp(lhs.micro_sec_since_epoch() + ms);
}

/***
 *
 * @param lhs
 * @param seconds
 * @return
 */
inline Timestamp operator+(const Timestamp& lhs, double seconds) {
    auto delta = static_cast<uint64_t>(seconds * Timestamp::k_micro_sec_per_sec);
    return Timestamp(lhs.micro_sec_since_epoch() + delta);
}

/***
 *
 * @param lhs
 * @param ms
 * @return
 */
inline Timestamp operator-(const Timestamp& lhs, uint64_t ms) {
    return Timestamp(lhs.micro_sec_since_epoch() - ms);
}

/***
 *
 * @param lhs
 * @param seconds
 * @return
 */
inline Timestamp operator-(const Timestamp& lhs, double seconds) {
    auto delta = static_cast<uint64_t>(seconds * Timestamp::k_micro_sec_per_sec);
    return Timestamp(lhs.micro_sec_since_epoch() - delta);
}

/***
 *
 * @param high
 * @param low
 * @return
 */
inline double operator-(const Timestamp& high, const Timestamp& low) {
    uint64_t diff = high.micro_sec_since_epoch() - low.micro_sec_since_epoch();
    return static_cast<double>(diff) / Timestamp::k_micro_sec_per_sec;
}
}
}

#endif //MMAISERVER_TIMESTAMP_H
