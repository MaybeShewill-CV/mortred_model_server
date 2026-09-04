/************************************************
 * Author: Codex
 * File: openapi_consistency_test.cc
 *
 * Verifies docs/openapi.json matches repository reality:
 * - a valid OpenAPI 3.0 document;
 * - every conf/server/<name>.toml server_uri appears in paths;
 * - model paths declare Bearer auth while health endpoints stay public;
 * - response components exist for all contract status codes.
 * Run from the repository root (see test/CMakeLists.txt WORKING_DIRECTORY).
 ************************************************/

#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <rapidjson/document.h>
#include <toml/toml.hpp>

namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<std::string> collect_server_uris() {
    std::vector<std::string> uris;
    for (const auto& entry : std::filesystem::recursive_directory_iterator("conf/server")) {
        if (!entry.is_regular_file() || entry.path().extension() != ".toml") {
            continue;
        }
        auto parsed = toml::parse_file(entry.path().string());
        if (!parsed) {
            continue;
        }
        auto table = std::move(parsed).table();
        for (const auto& [key, value] : table) {
            const auto* tbl = value.as_table();
            if (tbl == nullptr) {
                continue;
            }
            if (tbl->contains("server_uri")) {
                const auto uri = (*tbl)["server_uri"].value_or<std::string>("");
                if (!uri.empty()) {
                    uris.push_back(uri);
                }
            }
        }
    }
    return uris;
}

}  // namespace

TEST(openapi_consistency, document_is_valid_openapi_3) {
    rapidjson::Document doc;
    auto text = read_file("docs/openapi.json");
    ASSERT_FALSE(text.empty()) << "docs/openapi.json missing (run python scripts/gen_openapi.py)";
    ASSERT_FALSE(doc.Parse(text.c_str()).HasParseError());
    EXPECT_TRUE(doc.IsObject());
    EXPECT_STREQ(doc["openapi"].GetString(), "3.0.0");
    EXPECT_TRUE(doc.HasMember("info"));
    EXPECT_TRUE(doc.HasMember("paths"));
}

TEST(openapi_consistency, every_server_uri_is_declared) {
    rapidjson::Document doc;
    doc.Parse(read_file("docs/openapi.json").c_str());
    ASSERT_FALSE(doc.HasParseError());
    const auto& paths = doc["paths"];
    const auto uris = collect_server_uris();
    ASSERT_FALSE(uris.empty()) << "no conf/server/*.toml server_uri found";
    for (const auto& uri : uris) {
        EXPECT_TRUE(paths.HasMember(uri.c_str()))
            << "server_uri " << uri << " is missing from docs/openapi.json paths";
    }
}

TEST(openapi_consistency, model_paths_require_bearer_auth) {
    rapidjson::Document doc;
    doc.Parse(read_file("docs/openapi.json").c_str());
    ASSERT_FALSE(doc.HasParseError());
    const auto& paths = doc["paths"];
    const std::set<std::string> public_paths = {
        "/healthz", "/ready", "/metrics", "/openapi.json"};
    for (auto it = paths.MemberBegin(); it != paths.MemberEnd(); ++it) {
        const std::string path = it->name.GetString();
        if (public_paths.count(path) != 0) {
            continue;
        }
        ASSERT_TRUE(it->value.HasMember("post"))
            << "model path must be POST-only: " << path;
        const auto& post = it->value["post"];
        EXPECT_TRUE(post.HasMember("security")) << "model path missing security: " << path;
        if (post.HasMember("security")) {
            const auto& security = post["security"];
            ASSERT_TRUE(security.IsArray() && security.Size() > 0);
            EXPECT_TRUE(security[0].HasMember("bearerAuth"))
                << "model path must use bearerAuth: " << path;
        }
    }
}

TEST(openapi_consistency, contract_status_codes_are_covered) {
    rapidjson::Document doc;
    doc.Parse(read_file("docs/openapi.json").c_str());
    ASSERT_FALSE(doc.HasParseError());

    EXPECT_TRUE(doc.HasMember("components"));
    EXPECT_TRUE(doc["components"].HasMember("securitySchemes"));
    EXPECT_TRUE(doc["components"]["securitySchemes"].HasMember("bearerAuth"));

    const auto& responses = doc["components"]["responses"];
    const std::vector<const char*> required = {
        "BadRequest", "Unauthorized", "NotFound", "MethodNotAllowed", "PayloadTooLarge",
        "UnsupportedMediaType", "ValidationError", "RateLimited", "InternalError", "NotReady",
        "GatewayTimeout"};
    for (const char* name : required) {
        EXPECT_TRUE(responses.HasMember(name)) << "missing response component: " << name;
    }
}

TEST(openapi_consistency, unified_envelope_schemas_are_declared) {
    rapidjson::Document doc;
    doc.Parse(read_file("docs/openapi.json").c_str());
    ASSERT_FALSE(doc.HasParseError());
    const auto& schemas = doc["components"]["schemas"];

    ASSERT_TRUE(schemas.HasMember("UnifiedResponse"));
    const auto& envelope = schemas["UnifiedResponse"];
    ASSERT_TRUE(envelope["properties"].HasMember("status"));
    ASSERT_TRUE(envelope["properties"].HasMember("task_id"));
    ASSERT_TRUE(envelope["properties"].HasMember("results"));
    ASSERT_TRUE(envelope["properties"].HasMember("partial"));
    ASSERT_TRUE(envelope["properties"].HasMember("errors"));

    ASSERT_TRUE(schemas.HasMember("ResponseItem"));
    ASSERT_TRUE(schemas.HasMember("ResponseError"));
    ASSERT_TRUE(schemas.HasMember("OutputOptions"));
    // the legacy request shape must not survive in the generated document
    EXPECT_FALSE(schemas.HasMember("ImgRequest"));
}

TEST(openapi_consistency, every_model_request_requires_images_and_is_strict) {
    rapidjson::Document doc;
    doc.Parse(read_file("docs/openapi.json").c_str());
    ASSERT_FALSE(doc.HasParseError());
    const auto& schemas = doc["components"]["schemas"];
    int checked = 0;
    for (auto it = schemas.MemberBegin(); it != schemas.MemberEnd(); ++it) {
        const std::string name = it->name.GetString();
        if (name.rfind("Request_", 0) != 0) {
            continue;
        }
        const auto& schema = it->value;
        ASSERT_TRUE(schema.HasMember("required"));
        const auto& required = schema["required"];
        ASSERT_TRUE(required.IsArray());
        bool has_images = false;
        for (const auto& entry : required.GetArray()) {
            if (std::string(entry.GetString()) == "images") {
                has_images = true;
            }
        }
        EXPECT_TRUE(has_images) << "model request schema must require images[]: " << name;
        // strict envelope: unknown keys are rejected with 422, so additionalProperties
        // must be false on every generated request schema
        ASSERT_TRUE(schema.HasMember("additionalProperties"));
        EXPECT_FALSE(schema["additionalProperties"].GetBool())
            << "request schema must reject unknown fields: " << name;
        checked++;
    }
    EXPECT_GT(checked, 0) << "no generated model request schemas found";
}

TEST(openapi_consistency, legacy_html_health_endpoints_are_gone) {
    rapidjson::Document doc;
    doc.Parse(read_file("docs/openapi.json").c_str());
    ASSERT_FALSE(doc.HasParseError());
    const auto& paths = doc["paths"];
    EXPECT_FALSE(paths.HasMember("/welcome"));
    EXPECT_FALSE(paths.HasMember("/hello_world"));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
