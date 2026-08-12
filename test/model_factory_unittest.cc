/************************************************
 * Author: Codex
 * File: model_factory_unittest.cc
 * Date: 2026-08-12
 ************************************************/

#include <memory>

#include <gtest/gtest.h>
#include <glog/logging.h>

#include "common/status_code.h"
#include "factory/base_factory.h"
#include "models/base_model.h"
#include "server/abstract_server.h"

struct fake_model_input {};
struct fake_model_output {};

class FakeModel : public jinq::models::BaseAiModel<fake_model_input, fake_model_output> {
  public:
    jinq::common::StatusCode init(const decltype(toml::parse(""))&) override {
        return jinq::common::StatusCode::OK;
    }
    jinq::common::StatusCode run(const fake_model_input&, fake_model_output&) override {
        return jinq::common::StatusCode::OK;
    }
    bool is_successfully_initialized() const override {
        return true;
    }
};

using FakeModelBase = jinq::models::BaseAiModel<fake_model_input, fake_model_output>;

class FakeServer : public jinq::server::BaseAiServer {
  public:
    jinq::common::StatusCode init(const decltype(toml::parse(""))&) override {
        return jinq::common::StatusCode::OK;
    }
    void serve_process(WFHttpTask*) override {}
    bool is_successfully_initialized() const override {
        return true;
    }
};

TEST(model_factory, register_and_get_model) {
    jinq::factory::ModelRegistrar<FakeModelBase, FakeModel> registrar("fake_model");

    auto model = jinq::factory::ModelFactory<FakeModelBase>::get_instance().get_model("fake_model");
    EXPECT_NE(model, nullptr);
    EXPECT_TRUE(model->is_successfully_initialized());

    auto again = jinq::factory::ModelFactory<FakeModelBase>::get_instance().get_model("fake_model");
    EXPECT_NE(again, nullptr);

    auto missing = jinq::factory::ModelFactory<FakeModelBase>::get_instance().get_model("no_such_model");
    EXPECT_EQ(missing, nullptr);
}

TEST(server_factory, register_and_get_server) {
    jinq::factory::ServerRegistrar<jinq::server::BaseAiServer, FakeServer> registrar("fake_server");

    auto server = jinq::factory::ServerFactory<jinq::server::BaseAiServer>::get_instance().get_server("fake_server");
    EXPECT_NE(server, nullptr);
    EXPECT_TRUE(server->is_successfully_initialized());

    auto missing = jinq::factory::ServerFactory<jinq::server::BaseAiServer>::get_instance().get_server("no_such_server");
    EXPECT_EQ(missing, nullptr);
}
