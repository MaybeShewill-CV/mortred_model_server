/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: subject.h
* Date: 26-9-15
************************************************/

// Source-subject derivation for the gateway edge rate limiter (PR-2 of the
// rate-limiting plan; the metering kernel is gcra.h / PR-1, the gateway
// wiring is PR-3b). Pure functions, no workflow/toml/metrics dependencies.
//
// One question is answered here: WHICH source identity does a request count
// against? The rule set is the anti-spoofing core of the whole limiter —
// every clause below exists because a naive version of it is exploitable:
//
//   1. The TCP peer is the anchor. If the peer is NOT a configured trusted
//      proxy, every forwarding header is IGNORED and the peer itself is the
//      subject: an untrusted client must not choose its own bucket by
//      writing X-Forwarded-For.
//   2. If the peer IS trusted, the candidate chain comes from the
//      Forwarded header (RFC 7239) when it carries for= values, else from
//      X-Forwarded-For. The chain is walked RIGHT to LEFT (the rightmost
//      entry was appended by the proxy closest to us and is the most
//      trustworthy); trusted-proxy hops are skipped; the first non-trusted
//      parseable address is the subject.
//   3. An unparseable rightmost hop ("for=unknown", obfuscated tokens)
//      FALLS BACK TO THE PEER bucket instead of skipping left: skipping
//      would let a client hide behind an opaque token. Aggregating on the
//      proxy bucket is the conservative direction.
//   4. Fully trusted chains and missing headers also fall back to the peer.
//   5. An unparseable peer goes to a single shared "unknown" bucket
//      (family 0): garbage peers must rate-limit together, never crash.
//
// Trust is EXACT-ADDRESS only (v4 /32, v6 /128) — never a prefix: trusting
// a prefix would let anyone inside it mint trusted-proxy status.
//
// Folding into SubjectKey (the metering identity):
//   IPv4            -> /32   (prefix[0..3],   prefix[4..7] zero)
//   IPv6            -> /64   (prefix[0..7])   — a user rotating inside
//                        their /64 must not mint fresh buckets
//   v4-mapped (::ffff:a.b.c.d) -> IPv4 (both in peers and header values)

#ifndef MORTRED_CONTROL_RATE_LIMIT_SUBJECT_H
#define MORTRED_CONTROL_RATE_LIMIT_SUBJECT_H

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "control/rate_limit/gcra.h"

namespace mortred {
namespace control {
namespace ratelimit {

/*** A fully-parsed address. family 0 means "did not parse" — distinct from
 * SubjectKey.family 0, which is the shared unknown METERING bucket. */
struct IpAddr {
    uint8_t family = 0;  // 0 invalid, 4, 6
    std::array<uint8_t, 16> bytes{};

    bool valid() const { return family == 4 || family == 6; }
};

inline bool is_hex_digit(char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

inline uint32_t hex_value(char c) {
    if (c >= '0' && c <= '9') {
        return static_cast<uint32_t>(c - '0');
    }
    if (c >= 'a' && c <= 'f') {
        return static_cast<uint32_t>(c - 'a' + 10);
    }
    return static_cast<uint32_t>(c - 'A' + 10);
}

inline bool is_dec_digit(char c) { return c >= '0' && c <= '9'; }

inline std::string_view trim_view(std::string_view s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t')) {
        ++b;
    }
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t')) {
        --e;
    }
    return s.substr(b, e - b);
}

inline bool iequal(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        const char x = a[i] | 0x20;
        const char y = b[i] | 0x20;
        if ((x < 'a' || x > 'z') ? a[i] != b[i] : x != y) {
            return false;
        }
    }
    return true;
}

/*** Strict dotted-quad IPv4. Leading zeros accepted (inet_aton-compatible);
 * anything else — wrong group count, >255, empty groups, trailing junk,
 * zone ids — fails. */
inline IpAddr parse_ipv4(std::string_view s) {
    IpAddr out;
    if (s.empty() || s.size() > 15) {
        return out;
    }
    uint32_t parts[4] = {0, 0, 0, 0};
    size_t i = 0;
    for (int p = 0; p < 4; ++p) {
        if (i >= s.size() || !is_dec_digit(s[i])) {
            return out;
        }
        uint32_t value = 0;
        int digits = 0;
        while (i < s.size() && is_dec_digit(s[i])) {
            value = value * 10 + static_cast<uint32_t>(s[i] - '0');
            ++i;
            ++digits;
            if (digits > 3 || value > 255) {
                return out;
            }
        }
        parts[p] = value;
        if (p < 3) {
            if (i >= s.size() || s[i] != '.') {
                return out;
            }
            ++i;
        }
    }
    if (i != s.size()) {
        return out;
    }
    out.family = 4;
    out.bytes[0] = static_cast<uint8_t>(parts[0]);
    out.bytes[1] = static_cast<uint8_t>(parts[1]);
    out.bytes[2] = static_cast<uint8_t>(parts[2]);
    out.bytes[3] = static_cast<uint8_t>(parts[3]);
    return out;
}

namespace detail {

/*** parse one colon-separated half of an IPv6 literal into groups.
 * last_may_be_v4: the final segment may be an embedded dotted quad (counts
 * as two groups). Returns group count or -1 on error. */
inline int parse_v6_groups(std::string_view part, std::array<uint16_t, 8>& groups,
                           bool last_may_be_v4) {
    if (part.empty()) {
        return 0;
    }
    if (last_may_be_v4) {
        const size_t last_colon = part.rfind(':');
        const auto tail = part.substr(
            last_colon == std::string_view::npos ? 0 : last_colon + 1);
        if (tail.find('.') != std::string_view::npos) {
            const IpAddr v4 = parse_ipv4(tail);
            if (!v4.valid()) {
                return -1;
            }
            std::array<uint16_t, 8> head_groups{};
            const int head =
                (last_colon == std::string_view::npos)
                    ? 0
                    : parse_v6_groups(part.substr(0, last_colon), head_groups, false);
            if (head < 0 || head + 2 > 8) {
                return -1;
            }
            for (int i = 0; i < head; ++i) {
                groups[i] = head_groups[i];
            }
            groups[head] = static_cast<uint16_t>((v4.bytes[0] << 8) | v4.bytes[1]);
            groups[head + 1] = static_cast<uint16_t>((v4.bytes[2] << 8) | v4.bytes[3]);
            return head + 2;
        }
    }
    int count = 0;
    size_t i = 0;
    while (i < part.size()) {
        const size_t seg_start = i;
        while (i < part.size() && is_hex_digit(part[i])) {
            ++i;
        }
        const auto seg = part.substr(seg_start, i - seg_start);
        if (seg.empty() || seg.size() > 4) {
            return -1;
        }
        uint32_t value = 0;
        for (char c : seg) {
            value = value * 16 + hex_value(c);
        }
        if (count >= 8) {
            return -1;
        }
        groups[count] = static_cast<uint16_t>(value);
        ++count;
        if (i == part.size()) {
            return count;
        }
        if (part[i] != ':') {
            return -1;  // stray '.', '%', ... land here
        }
        ++i;
        if (i == part.size()) {
            return -1;  // trailing single colon
        }
    }
    return -1;
}

}  // namespace detail

/*** Strict IPv6 with :: compression and embedded v4 tails. Zone ids
 * (fe80::1%eth0) are rejected: scoped addresses are meaningless as metering
 * identities. ::ffff:a.b.c.d normalizes to family 4. */
inline IpAddr parse_ipv6(std::string_view s) {
    IpAddr out;
    if (s.empty() || s.size() > 45) {
        return out;
    }
    const size_t compress = s.find("::");
    if (compress != std::string_view::npos &&
        s.find("::", compress + 1) != std::string_view::npos) {
        return out;  // more than one ::
    }
    std::array<uint16_t, 8> groups{};
    int left = 0;
    int right = 0;
    std::array<uint16_t, 8> right_groups{};
    if (compress == std::string_view::npos) {
        left = detail::parse_v6_groups(s, groups, true);
        if (left != 8) {
            return out;
        }
    } else {
        const std::string_view head = s.substr(0, compress);
        const std::string_view tail = s.substr(compress + 2);
        left = detail::parse_v6_groups(head, groups, false);
        right = detail::parse_v6_groups(tail, right_groups, true);
        if (left < 0 || right < 0 || left + right > 7) {
            return out;  // with :: the total must leave at least one zero group
        }
        for (int i = 0; i < right; ++i) {
            groups[8 - right + i] = right_groups[i];
        }
    }
    out.family = 6;
    for (int g = 0; g < 8; ++g) {
        out.bytes[2 * g] = static_cast<uint8_t>(groups[g] >> 8);
        out.bytes[2 * g + 1] = static_cast<uint8_t>(groups[g] & 0xff);
    }
    // v4-mapped (::ffff:0:0/96) folds to plain IPv4
    bool mapped = true;
    for (size_t b = 0; b < 10 && mapped; ++b) {
        if (out.bytes[b] != 0) {
            mapped = false;
        }
    }
    if (mapped && out.bytes[10] == 0xff && out.bytes[11] == 0xff) {
        IpAddr v4;
        v4.family = 4;
        v4.bytes[0] = out.bytes[12];
        v4.bytes[1] = out.bytes[13];
        v4.bytes[2] = out.bytes[14];
        v4.bytes[3] = out.bytes[15];
        return v4;
    }
    return out;
}

/*** Parse an IP literal: anything containing ':' goes to the v6 parser
 * (after header normalization ports are already stripped), else v4. */
inline IpAddr parse_ip(std::string_view s) {
    s = trim_view(s);
    if (s.find(':') != std::string_view::npos) {
        return parse_ipv6(s);
    }
    return parse_ipv4(s);
}

/*** Exact-address trusted-proxy set. Trust is never prefix-wide (see file
 * header). Invalid entries in the config list are skipped — a typo must not
 * disable or poison the whole trust chain silently at parse time; the
 * gateway wiring layer warns about skipped entries. */
struct TrustedProxies {
    std::vector<IpAddr> entries;

    static TrustedProxies parse(const std::vector<std::string>& ips) {
        TrustedProxies out;
        out.entries.reserve(ips.size());
        for (const auto& ip : ips) {
            const IpAddr addr = parse_ip(ip);
            if (addr.valid()) {
                out.entries.push_back(addr);
            }
        }
        return out;
    }

    bool empty() const { return entries.empty(); }

    bool contains(const IpAddr& addr) const {
        if (!addr.valid()) {
            return false;
        }
        for (const auto& entry : entries) {
            if (entry.family == addr.family && entry.bytes == addr.bytes) {
                return true;
            }
        }
        return false;
    }
};

/*** Fold a parsed address into the metering identity (v4 /32, v6 /64). */
inline SubjectKey fold_subject(const IpAddr& addr) {
    SubjectKey key;
    key.family = addr.family;
    const size_t n = addr.family == 4 ? 4 : 8;
    for (size_t i = 0; i < n; ++i) {
        key.prefix[i] = addr.bytes[i];
    }
    return key;
}

namespace detail {

/*** Header-value normalization: strip "[...]" brackets and a trailing
 * ":port". A bare IPv6 has >= 2 colons, so exactly one colon means v4:port. */
inline std::string_view strip_brackets_and_port(std::string_view s) {
    if (!s.empty() && s.front() == '[') {
        const size_t close = s.find(']');
        if (close != std::string_view::npos) {
            return s.substr(1, close - 1);
        }
        return s;  // malformed bracket: let parse_ip reject it
    }
    size_t colons = 0;
    for (char c : s) {
        if (c == ':') {
            ++colons;
        }
    }
    if (colons == 1) {
        return s.substr(0, s.find(':'));
    }
    return s;
}

inline void collect_xff(std::string_view header, std::vector<std::string_view>& out) {
    size_t pos = 0;
    while (pos <= header.size()) {
        const size_t comma = header.find(',', pos);
        const auto entry = trim_view(header.substr(
            pos, comma == std::string_view::npos ? std::string_view::npos : comma - pos));
        if (!entry.empty()) {
            out.push_back(entry);
        }
        if (comma == std::string_view::npos) {
            break;
        }
        pos = comma + 1;
    }
}

/*** Forwarded (RFC 7239): comma-separated elements, semicolon-separated
 * params; take the first for= value of each element, unquote it. Param
 * names are case-insensitive. */
inline void collect_forwarded_for(std::string_view header,
                                  std::vector<std::string_view>& out) {
    size_t pos = 0;
    while (pos <= header.size()) {
        const size_t comma = header.find(',', pos);
        const auto element = header.substr(
            pos, comma == std::string_view::npos ? std::string_view::npos : comma - pos);
        size_t epos = 0;
        while (epos <= element.size()) {
            const size_t semi = element.find(';', epos);
            const auto param = trim_view(element.substr(
                epos, semi == std::string_view::npos ? std::string_view::npos : semi - epos));
            if (param.size() >= 4 && iequal(param.substr(0, 4), "for=")) {
                auto value = trim_view(param.substr(4));
                if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
                    value = value.substr(1, value.size() - 2);
                }
                value = trim_view(value);
                if (!value.empty()) {
                    out.push_back(value);
                }
                break;
            }
            if (semi == std::string_view::npos) {
                break;
            }
            epos = semi + 1;
        }
        if (comma == std::string_view::npos) {
            break;
        }
        pos = comma + 1;
    }
}

}  // namespace detail

/*** Derive the metering subject for one request. See the file header for
 * the rule set; the two security-critical properties are (1) an untrusted
 * peer's headers are ignored entirely and (2) an unparseable rightmost hop
 * falls back to the peer bucket instead of skipping left.
 *
 * used_header (optional): whether a Forwarded/XFF value produced the
 * subject — the wiring layer logs this with rejection samples. */
inline SubjectKey derive_subject(const std::string_view peer_text,
                                 const TrustedProxies& trusted,
                                 const std::string_view forwarded_header,
                                 const std::string_view xff_header,
                                 bool* used_header = nullptr) {
    if (used_header != nullptr) {
        *used_header = false;
    }
    const IpAddr peer = parse_ip(peer_text);
    if (!peer.valid()) {
        SubjectKey unknown;
        unknown.family = 0;  // shared bucket; must differ from default (4),
                             // or it would collide with 0.0.0.0
        return unknown;
    }
    if (!trusted.contains(peer)) {
        return fold_subject(peer);
    }
    std::vector<std::string_view> chain;
    if (!forwarded_header.empty()) {
        detail::collect_forwarded_for(forwarded_header, chain);
    }
    if (chain.empty() && !xff_header.empty()) {
        detail::collect_xff(xff_header, chain);
    }
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
        const IpAddr candidate = parse_ip(detail::strip_brackets_and_port(*it));
        if (!candidate.valid()) {
            break;  // unattributable hop: conservative fallback to the peer
        }
        if (!trusted.contains(candidate)) {
            if (used_header != nullptr) {
                *used_header = true;
            }
            return fold_subject(candidate);
        }
        // trusted hop closer to us: keep walking left
    }
    return fold_subject(peer);
}

}  // namespace ratelimit
}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_RATE_LIMIT_SUBJECT_H
