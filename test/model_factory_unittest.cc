/************************************************
 * Author: Codex
 * File: model_factory_unittest.cc
 * Date: 2026-08-12
 ************************************************/

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <glog/logging.h>

#include "common/status_code.h"
#include "factory/base_factory.h"
#include "models/base_model.h"

using jinq::common::StatusCode;

struct fake_model_input {};
struct fake_model_output {};

// test-only polymorphic base: tag() verifies the concrete type the factory created
class FakeModelBase : public jinq::models::BaseAiModel<fake_model_input, fake_model_output> {
public:
    ~FakeModelBase() override = default;
    virtual std::string tag() const = 0;
};

class FakeModel : public FakeModelBase {
public:
    StatusCode init(const toml::table&) override {
        return StatusCode::OK;
    }
    StatusCode run_impl(const fake_model_input&, fake_model_output&) override {
        return StatusCode::OK;
    }
    bool is_successfully_initialized() const override {
        return true;
    }
    std::string tag() const override {
        return "FakeModel";
    }
};

class FakeModel2 : public FakeModelBase {
public:
    StatusCode init(const toml::table&) override {
        return StatusCode::OK;
    }
    StatusCode run_impl(const fake_model_input&, fake_model_output&) override {
        return StatusCode::OK;
    }
    bool is_successfully_initialized() const override {
        return true;
    }
    std::string tag() const override {
        return "FakeModel2";
    }
};

class FakeServer : public jinq::server::BaseAiServer {
public:
    StatusCode init(const toml::table&) override {
        return StatusCode::OK;
    }
    void serve_process(WFHttpTask*) override {}
    bool is_successfully_initialized() const override {
        return true;
    }
};

// differs from FakeServer by is_successfully_initialized() for type verification
class FakeServer2 : public jinq::server::BaseAiServer {
public:
    StatusCode init(const toml::table&) override {
        return StatusCode::OK;
    }
    void serve_process(WFHttpTask*) override {}
    bool is_successfully_initialized() const override {
        return false;
    }
};

using ModelFactory = jinq::factory::ModelFactory<FakeModelBase>;
using ServerFactory = jinq::factory::ServerFactory<jinq::server::BaseAiServer>;

TEST(model_factory, register_and_create_model) {
    ModelFactory::get_instance().register_type<FakeModel>("fake_model");

    auto model = ModelFactory::get_instance().create("fake_model");
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->tag(), "FakeModel");
}

// regression: the factory must still create by name after the registering scope
// closes (the old stack-local registrar left a dangling pointer here)
TEST(model_factory, create_stays_valid_after_registration_scope_closes) {
    {
        ModelFactory::get_instance().register_type<FakeModel>("scoped_model");
        auto first = ModelFactory::get_instance().create("scoped_model");
        ASSERT_NE(first, nullptr);
    }

    auto later = ModelFactory::get_instance().create("scoped_model");
    ASSERT_NE(later, nullptr);
    EXPECT_EQ(later->tag(), "FakeModel");
}

TEST(model_factory, re_registering_same_name_replaces_creator) {
    auto& factory = ModelFactory::get_instance();
    factory.register_type<FakeModel>("overwrite_me");
    factory.register_type<FakeModel2>("overwrite_me");

    auto model = factory.create("overwrite_me");
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->tag(), "FakeModel2");
}

TEST(model_factory, create_unknown_name_returns_nullptr) {
    EXPECT_EQ(ModelFactory::get_instance().create("no_such_model"), nullptr);
}

TEST(model_factory, empty_name_is_rejected) {
    auto& factory = ModelFactory::get_instance();
    factory.register_type<FakeModel>("");
    EXPECT_EQ(factory.create(""), nullptr);
}

// smoke test: concurrent register/create must be thread-safe and type-correct
TEST(model_factory, concurrent_register_and_create) {
    constexpr int k_threads = 8;
    constexpr int k_iterations = 100;
    std::atomic<int> failures{0};

    std::vector<std::thread> threads;
    threads.reserve(k_threads);
    for (int t = 0; t < k_threads; ++t) {
        threads.emplace_back([t, &failures]() {
            auto& factory = ModelFactory::get_instance();
            for (int i = 0; i < k_iterations; ++i) {
                const std::string name = "concurrent_" + std::to_string(t) + "_" + std::to_string(i);
                factory.register_type<FakeModel>(name);
                auto model = factory.create(name);
                if (model == nullptr || model->tag() != "FakeModel") {
                    ++failures;
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(failures.load(), 0);
}

// creator-closure registration: spec-driven servers register arbitrary lambdas
// instead of named concrete classes
TEST(model_factory, register_creator_builds_custom_closure) {
    auto& factory = ModelFactory::get_instance();
    factory.register_creator("closure_model", []() -> std::unique_ptr<FakeModelBase> {
        return std::unique_ptr<FakeModelBase>(new FakeModel2());
    });

    auto model = factory.create("closure_model");
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->tag(), "FakeModel2");
}

// creator closures follow the same overwrite-on-same-name semantics as register_type
TEST(model_factory, register_creator_replaces_previous_registration) {
    auto& factory = ModelFactory::get_instance();
    factory.register_creator("closure_overwrite", []() -> std::unique_ptr<FakeModelBase> {
        return std::unique_ptr<FakeModelBase>(new FakeModel());
    });
    factory.register_creator("closure_overwrite", []() -> std::unique_ptr<FakeModelBase> {
        return std::unique_ptr<FakeModelBase>(new FakeModel2());
    });

    auto model = factory.create("closure_overwrite");
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->tag(), "FakeModel2");
}

TEST(model_factory, register_creator_rejects_empty_name_and_null_closure) {
    auto& factory = ModelFactory::get_instance();
    factory.register_creator("", []() -> std::unique_ptr<FakeModelBase> {
        return std::unique_ptr<FakeModelBase>(new FakeModel());
    });
    EXPECT_EQ(factory.create(""), nullptr);

    factory.register_creator("null_closure", nullptr);
    EXPECT_EQ(factory.create("null_closure"), nullptr);
}

TEST(server_factory, register_and_create_server) {
    ServerFactory::get_instance().register_type<FakeServer>("fake_server");

    auto server = ServerFactory::get_instance().create("fake_server");
    ASSERT_NE(server, nullptr);
    EXPECT_TRUE(server->is_successfully_initialized());
}

TEST(server_factory, create_stays_valid_after_registration_scope_closes) {
    {
        ServerFactory::get_instance().register_type<FakeServer>("scoped_server");
        auto first = ServerFactory::get_instance().create("scoped_server");
        ASSERT_NE(first, nullptr);
    }

    auto later = ServerFactory::get_instance().create("scoped_server");
    ASSERT_NE(later, nullptr);
    EXPECT_TRUE(later->is_successfully_initialized());
}

TEST(server_factory, re_registering_same_name_replaces_creator) {
    auto& factory = ServerFactory::get_instance();
    factory.register_type<FakeServer>("server_overwrite");
    factory.register_type<FakeServer2>("server_overwrite");

    auto server = factory.create("server_overwrite");
    ASSERT_NE(server, nullptr);
    // FakeServer2 returns false, proving the later registration won
    EXPECT_FALSE(server->is_successfully_initialized());
}

TEST(server_factory, create_unknown_name_returns_nullptr) {
    EXPECT_EQ(ServerFactory::get_instance().create("no_such_server"), nullptr);
}
