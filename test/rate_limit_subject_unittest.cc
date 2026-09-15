/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: rate_limit_subject_unittest.cc
* Date: 26-9-15
************************************************/

// Source-subject derivation tests. The deception matrix is the whole point
// of this module: every case encodes one spoofing attempt that the naive
// implementation would fall for (headers honored from untrusted peers,
// leftward skipping past opaque tokens, prefix-wide trust, port/bracket
// confusion, v4-mapped bypasses of the /64 fold).

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "control/rate_limit/subject.h"

namespace {

using mortred::control::ratelimit::IpAddr;
using mortred::control::ratelimit::SubjectKey;
using mortred::control::ratelimit::TrustedProxies;
using mortred::control::ratelimit::derive_subject;
using mortred::control::ratelimit::fold_subject;
using mortred::control::ratelimit::parse_ip;

SubjectKey k4(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
    SubjectKey k;
    k.family = 4;
    k.prefix = {a, b, c, d, 0, 0, 0, 0};
    return k;
}

TrustedProxies loopback() {
    return TrustedProxies::parse({"127.0.0.1", "::1"});
}

}  // namespace

TEST(ParseIp, ValidIpv4) {
    EXPECT_TRUE(parse_ip("1.2.3.4").valid());
    EXPECT_EQ(parse_ip("1.2.3.4").family, 4);
    EXPECT_TRUE(parse_ip("0.0.0.0").valid());
    EXPECT_TRUE(parse_ip("255.255.255.255").valid());
    EXPECT_TRUE(parse_ip("010.1.1.1").valid());  // inet_aton-style zeros
    const IpAddr a = parse_ip("192.168.1.100");
    EXPECT_EQ(a.bytes[0], 192);
    EXPECT_EQ(a.bytes[3], 100);
}

TEST(ParseIp, InvalidIpv4) {
    for (const char* bad : {"1.2.3", "1.2.3.4.5", "256.1.1.1", "1.2.3.", ".1.2.3",
                            "a.b.c.d", "", "1..2.3", "1.2.3.4x", "1.2.3.-4"}) {
        EXPECT_FALSE(parse_ip(bad).valid()) << bad;
    }
    // surrounding whitespace is trimmed, not an error
    EXPECT_TRUE(parse_ip(" 1.2.3.4 ").valid());
}

TEST(ParseIp, ValidIpv6Forms) {
    EXPECT_TRUE(parse_ip("::").valid());
    EXPECT_TRUE(parse_ip("::1").valid());
    EXPECT_TRUE(parse_ip("2001:db8::1").valid());
    EXPECT_TRUE(parse_ip("2001:0DB8::ABCD").valid());   // case-insensitive
    EXPECT_TRUE(parse_ip("1::").valid());
    EXPECT_TRUE(parse_ip("1::2").valid());
    EXPECT_TRUE(parse_ip("fe80::").valid());
    EXPECT_TRUE(parse_ip("64:ff9b::1.2.3.4").valid());  // embedded v4 tail
    EXPECT_TRUE(parse_ip("1:2:3:4:5:6:7:8").valid());   // full form
    EXPECT_TRUE(parse_ip("1:2:3:4:5:6:1.2.3.4").valid());
}

TEST(ParseIp, InvalidIpv6Forms) {
    for (const char* bad : {"1:2:3", "1::2::3", "12345::", ":",
                            "1:2:3:4:5:6:7:8:9", "1:2:3:4:5:6:7:8:",
                            "fe80::1%eth0", "1:2:3:4:5:6:7:1.2.3.4",
                            "::ffff:300.1.1.1", "2001:db8:::1"}) {
        EXPECT_FALSE(parse_ip(bad).valid()) << bad;
    }
}

TEST(ParseIp, V4MappedNormalizesToIpv4) {
    const IpAddr mapped = parse_ip("::ffff:8.8.8.8");
    ASSERT_TRUE(mapped.valid());
    EXPECT_EQ(mapped.family, 4);
    EXPECT_EQ(mapped.bytes[0], 8);
    EXPECT_EQ(mapped.bytes[3], 8);
    // a non-mapped v6 stays v6
    EXPECT_EQ(parse_ip("::1:2").family, 6);
}

TEST(FoldSubject, V4FoldsTo32) {
    const SubjectKey k = fold_subject(parse_ip("203.0.113.7"));
    EXPECT_EQ(k.family, 4);
    EXPECT_EQ(k.prefix[0], 203);
    EXPECT_EQ(k.prefix[3], 7);
    for (size_t i = 4; i < 8; ++i) {
        EXPECT_EQ(k.prefix[i], 0);
    }
}

TEST(FoldSubject, V6FoldsTo64AndIgnoresLowBits) {
    const SubjectKey k = fold_subject(parse_ip("2001:db8:1:2:dead:beef::1"));
    EXPECT_EQ(k.family, 6);
    const uint8_t want[8] = {0x20, 0x01, 0x0d, 0xb8, 0x00, 0x01, 0x00, 0x02};
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(k.prefix[i], want[i]) << "byte " << i;
    }
    // /64 siblings share one bucket
    EXPECT_EQ(fold_subject(parse_ip("2001:db8:1:2::1")),
              fold_subject(parse_ip("2001:db8:1:2:ffff:ffff:ffff:ffff")));
    // different /64 differs
    EXPECT_NE(fold_subject(parse_ip("2001:db8:1:2::1")),
              fold_subject(parse_ip("2001:db8:1:3::1")));
}

TEST(FoldSubject, V4MappedFoldsAsV4) {
    EXPECT_EQ(fold_subject(parse_ip("::ffff:9.9.9.9")), k4(9, 9, 9, 9));
}

TEST(TrustedProxies, ParseSkipsInvalidEntries) {
    const auto t = TrustedProxies::parse({"127.0.0.1", "not-an-ip", "::1", ""});
    EXPECT_EQ(t.entries.size(), 2u);
    EXPECT_TRUE(t.contains(parse_ip("127.0.0.1")));
    EXPECT_TRUE(t.contains(parse_ip("::1")));
    EXPECT_FALSE(t.contains(parse_ip("127.0.0.2")));
}

TEST(TrustedProxies, V4MappedMatchesV4Entry) {
    const auto t = TrustedProxies::parse({"127.0.0.1"});
    EXPECT_TRUE(t.contains(parse_ip("::ffff:127.0.0.1")));
}

// ---------------------------------------------------------------------------
// The deception matrix
// ---------------------------------------------------------------------------

TEST(DeriveSubject, UntrustedPeerIgnoresXff) {
    // THE core anti-spoofing case: public peer claims to be someone else
    const auto subject =
        derive_subject("203.0.113.9", loopback(), "", "1.1.1.1, 2.2.2.2");
    EXPECT_EQ(subject, k4(203, 0, 113, 9));
}

TEST(DeriveSubject, UntrustedPeerIgnoresForwarded) {
    const auto subject =
        derive_subject("203.0.113.9", loopback(), "for=1.1.1.1", "for=2.2.2.2");
    EXPECT_EQ(subject, k4(203, 0, 113, 9));
}

TEST(DeriveSubject, TrustedPeerNoHeadersUsesPeer) {
    const auto subject = derive_subject("127.0.0.1", loopback(), "", "");
    EXPECT_EQ(subject, k4(127, 0, 0, 1));
}

TEST(DeriveSubject, TrustedPeerUsesXffClient) {
    const auto subject = derive_subject("127.0.0.1", loopback(), "", "9.9.9.9");
    EXPECT_EQ(subject, k4(9, 9, 9, 9));
}

TEST(DeriveSubject, RightmostNonTrustedWins) {
    // two untrusted entries: the RIGHT one (appended by our proxy) wins
    const auto subject =
        derive_subject("127.0.0.1", loopback(), "", "6.6.6.6, 7.7.7.7");
    EXPECT_EQ(subject, k4(7, 7, 7, 7));
}

TEST(DeriveSubject, WalksLeftPastTrustedHops) {
    // chain client -> trusted proxy -> trusted proxy(=our peer)
    const auto subject =
        derive_subject("127.0.0.1", loopback(), "", "6.6.6.6, 127.0.0.1, ::1");
    EXPECT_EQ(subject, k4(6, 6, 6, 6));
}

TEST(DeriveSubject, AllTrustedChainFallsBackToPeer) {
    const auto subject =
        derive_subject("127.0.0.1", loopback(), "", "127.0.0.1, ::1");
    EXPECT_EQ(subject, k4(127, 0, 0, 1));
}

TEST(DeriveSubject, ForwardedWinsOverXff) {
    const auto subject =
        derive_subject("127.0.0.1", loopback(), "for=5.5.5.5", "6.6.6.6");
    EXPECT_EQ(subject, k4(5, 5, 5, 5));
}

TEST(DeriveSubject, ForwardedParamsAndQuoting) {
    // quoted value, extra params, multiple elements
    const auto subject = derive_subject(
        "127.0.0.1", loopback(),
        "for=\"1.2.3.4\";proto=http;host=x, for=7.7.7.7;by=127.0.0.1", "");
    EXPECT_EQ(subject, k4(7, 7, 7, 7));
}

TEST(DeriveSubject, ForwardedCaseInsensitiveParamAndCaselessHex) {
    const auto subject = derive_subject("127.0.0.1", loopback(),
                                        "FOR=[2001:DB8::1]:443", "");
    const SubjectKey want = fold_subject(parse_ip("2001:db8::1"));
    EXPECT_EQ(subject, want);
}

TEST(DeriveSubject, ForwardedBracketedV6PortStripped) {
    const auto subject =
        derive_subject("127.0.0.1", loopback(), "for=[2001:db8:1:2:a:b:c:d]:8443", "");
    const SubjectKey want = fold_subject(parse_ip("2001:db8:1:2:a:b:c:d"));
    EXPECT_EQ(subject, want);
}

TEST(DeriveSubject, XffPortStrippedFromV4) {
    const auto subject = derive_subject("127.0.0.1", loopback(), "", "1.2.3.4:5678");
    EXPECT_EQ(subject, k4(1, 2, 3, 4));
}

TEST(DeriveSubject, UnparseableRightmostFallsBackToPeer) {
    // "unknown" in the RIGHTMOST slot (appended by our trusted proxy, the
    // only slot an attacker cannot choose): the chain is unattributable,
    // skipping left would let a client hide behind an opaque token
    const auto subject =
        derive_subject("127.0.0.1", loopback(), "for=6.6.6.6, for=unknown", "");
    EXPECT_EQ(subject, k4(127, 0, 0, 1));
}

TEST(DeriveSubject, OpaqueTokenOnTheLeftNeverMatters) {
    // attacker-controlled LEFT slot: the rightmost non-trusted wins outright
    const auto subject =
        derive_subject("127.0.0.1", loopback(), "for=unknown, for=6.6.6.6", "");
    EXPECT_EQ(subject, k4(6, 6, 6, 6));
}

TEST(DeriveSubject, UnparseableXffEntryFallsBackToPeer) {
    const auto subject =
        derive_subject("127.0.0.1", loopback(), "", "6.6.6.6, not-an-ip");
    EXPECT_EQ(subject, k4(127, 0, 0, 1));
}

TEST(DeriveSubject, HeaderV4MappedTreatedAsV4) {
    const auto subject =
        derive_subject("127.0.0.1", loopback(), "", "::ffff:8.8.4.4");
    EXPECT_EQ(subject, k4(8, 8, 4, 4));
}

TEST(DeriveSubject, WhitespaceTolerated) {
    const auto subject =
        derive_subject("127.0.0.1", loopback(), "", "  1.2.3.4 ,  5.6.7.8  ");
    EXPECT_EQ(subject, k4(5, 6, 7, 8));
}

TEST(DeriveSubject, EmptyTrustedListTrustsNobody) {
    // trusted_proxies="" in config: every peer is untrusted, headers dead
    const TrustedProxies none;
    const auto subject = derive_subject("127.0.0.1", none, "for=9.9.9.9", "8.8.8.8");
    EXPECT_EQ(subject, k4(127, 0, 0, 1));
}

TEST(DeriveSubject, InvalidPeerGoesToUnknownBucket) {
    const auto subject = derive_subject("not-an-ip", loopback(), "for=9.9.9.9", "");
    EXPECT_EQ(subject.family, 0);
    bool used = true;
    derive_subject("garbage", loopback(), "", "9.9.9.9", &used);
    EXPECT_FALSE(used);
}

TEST(DeriveSubject, UsedHeaderFlagReflectsSource) {
    bool used = false;
    derive_subject("127.0.0.1", loopback(), "", "9.9.9.9", &used);
    EXPECT_TRUE(used);  // subject came from the header
    used = false;
    derive_subject("203.0.113.9", loopback(), "", "9.9.9.9", &used);
    EXPECT_FALSE(used);  // untrusted peer: header ignored
    used = false;
    derive_subject("127.0.0.1", loopback(), "", "", &used);
    EXPECT_FALSE(used);  // no header at all
}
