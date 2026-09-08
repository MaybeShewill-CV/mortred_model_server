#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "models/backend/session_io.h"

using jinq::common::StatusCode;
using jinq::models::backend::TensorInfo;
using jinq::models::backend::apply_configured_io_names;

namespace {

TensorInfo named_info(const std::string& name) {
    TensorInfo info;
    info.name = name;
    return info;
}

std::vector<std::string> names_of(const std::vector<TensorInfo>& infos) {
    std::vector<std::string> names;
    names.reserve(infos.size());
    for (const auto& info : infos) {
        names.push_back(info.name);
    }
    return names;
}

}  // namespace

TEST(SessionIoNames, EmptyNamesKeepsDiscoveryOrder) {
    std::vector<TensorInfo> infos = {named_info("z"), named_info("a")};
    std::string err;
    ASSERT_EQ(apply_configured_io_names({}, &infos, "output", &err), StatusCode::OK);
    EXPECT_EQ(names_of(infos), (std::vector<std::string>{"z", "a"}));
}

TEST(SessionIoNames, SelectsSubsetInConfiguredOrder) {
    std::vector<TensorInfo> infos = {named_info("a"), named_info("b"), named_info("c")};
    std::string err;
    ASSERT_EQ(apply_configured_io_names({"c", "a"}, &infos, "output", &err), StatusCode::OK)
        << err;
    EXPECT_EQ(names_of(infos), (std::vector<std::string>{"c", "a"}));
}

TEST(SessionIoNames, MissingNameFails) {
    std::vector<TensorInfo> infos = {named_info("a")};
    std::string err;
    EXPECT_EQ(apply_configured_io_names({"b"}, &infos, "output", &err),
              StatusCode::MODEL_INIT_FAILED);
    EXPECT_NE(err.find("not found"), std::string::npos) << err;
    EXPECT_EQ(names_of(infos), (std::vector<std::string>{"a"}));
}

TEST(SessionIoNames, DuplicateNameFails) {
    std::vector<TensorInfo> infos = {named_info("a"), named_info("b")};
    std::string err;
    EXPECT_EQ(apply_configured_io_names({"a", "a"}, &infos, "input", &err),
              StatusCode::MODEL_INIT_FAILED);
    EXPECT_NE(err.find("duplicated"), std::string::npos) << err;
}

TEST(SessionIoNames, EmptyConfiguredNameFails) {
    std::vector<TensorInfo> infos = {named_info("a")};
    std::string err;
    EXPECT_EQ(apply_configured_io_names({""}, &infos, "output", &err),
              StatusCode::MODEL_INIT_FAILED);
    EXPECT_NE(err.find("empty"), std::string::npos) << err;
}

TEST(SessionIoNames, NullInfosFails) {
    std::string err;
    EXPECT_EQ(apply_configured_io_names({"a"}, nullptr, "output", &err),
              StatusCode::MODEL_INIT_FAILED);
}
