/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: base64_unittest.cc
* Date: 22-6-2
************************************************/

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "common/base64.h"

using jinq::common::base64::decode;
using jinq::common::base64::encode;

TEST(base64, encode_rfc4648_vectors) {
    EXPECT_EQ(encode(""), "");
    EXPECT_EQ(encode("f"), "Zg==");
    EXPECT_EQ(encode("fo"), "Zm8=");
    EXPECT_EQ(encode("foo"), "Zm9v");
    EXPECT_EQ(encode("foob"), "Zm9vYg==");
    EXPECT_EQ(encode("fooba"), "Zm9vYmE=");
    EXPECT_EQ(encode("foobar"), "Zm9vYmFy");

    // raw pointer overload
    const unsigned char raw[] = {'f', 'o', 'o'};
    EXPECT_EQ(encode(raw, sizeof(raw)), "Zm9v");
    EXPECT_EQ(encode(static_cast<const void*>(nullptr), 0), "");
    EXPECT_EQ(encode(static_cast<const void*>(nullptr), 5), "");
}

TEST(base64, decode_rfc4648_vectors) {
    EXPECT_EQ(decode("Zg=="), "f");
    EXPECT_EQ(decode("Zm8="), "fo");
    EXPECT_EQ(decode("Zm9v"), "foo");
    EXPECT_EQ(decode("Zm9vYg=="), "foob");
    EXPECT_EQ(decode("Zm9vYmE="), "fooba");
    EXPECT_EQ(decode("Zm9vYmFy"), "foobar");
}

TEST(base64, decode_invalid_input) {
    EXPECT_TRUE(decode("").empty());
    EXPECT_TRUE(decode("a").empty());        // bad length
    EXPECT_TRUE(decode("abcde").empty());    // bad length
    EXPECT_TRUE(decode("Zm9$").empty());     // invalid character
    EXPECT_TRUE(decode("Zm=8").empty());     // padding in the middle
    EXPECT_TRUE(decode("====").empty());
    EXPECT_TRUE(decode("Zg").empty());       // padding stripped but wrong remainder
}

TEST(base64, round_trip_binary_and_large) {
    // binary payload covering every byte value
    std::vector<unsigned char> bytes(768);
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<unsigned char>(i);
    }
    std::string encoded = encode(bytes.data(), bytes.size());
    std::string decoded = decode(encoded);
    ASSERT_EQ(decoded.size(), bytes.size());
    EXPECT_EQ(std::string(decoded.begin(), decoded.end()),
              std::string(bytes.begin(), bytes.end()));

    // sizes around block boundaries
    for (size_t len = 1; len <= 10; ++len) {
        std::string payload(len, 'a');
        EXPECT_EQ(decode(encode(payload)), payload) << "round trip failed at len=" << len;
    }
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
