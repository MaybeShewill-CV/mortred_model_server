/************************************************
 * Author: Codex
 * File: diffusion_param_unittest.cc
 * Date: 2026-09-01
 ************************************************/

// Verifies the diffusion adapter's request-param plumbing without engines:
// fake samplers record the input they receive, so every override can be
// asserted against the config-template-plus-params resolution rule.
// init also seeds sample_size / step defaults from the sampler TOML; a
// missing spatial contract fails init so HTTP never serves a 0x0 template.

#include <gtest/gtest.h>

#include <string>

#include "factory/diffusion_model_adapter.h"
#include "models/backend/param_spec.h"
#include "toml/toml.hpp"

using jinq::common::StatusCode;
using jinq::factory::diffusion::DiffusionModelAdapter;
using jinq::models::backend::ParamSet;
using jinq::models::io_define::common_io::image_input;
using namespace jinq::models::io_define::diffusion;

namespace {

toml::table parse_toml(const std::string &content) {
    auto parsed = toml::parse(content);
    if (!parsed) {
        ADD_FAILURE() << "fixture toml parse failed";
        return toml::table{};
    }
    return std::move(parsed).table();
}

toml::table ddpm_cfg() {
    return parse_toml(R"toml(
[DDPM_SAMPLER]
sample_size = [128, 128]
timesteps = 1000
channels = 3
)toml");
}

toml::table ddim_cfg() {
    return parse_toml(R"toml(
[DDIM_SAMPLER]
sample_size = [256, 256]
total_timesteps = 1000
sample_steps = 100
channels = 3
eta = 1.0
)toml");
}

toml::table cls_cond_cfg() {
    return parse_toml(R"toml(
[DDIM_SAMPLER]
sample_size = [128, 128]
total_timesteps = 1000
sample_steps = 100
channels = 3
eta = 0.5
cls_id = 3
)toml");
}

toml::table ldm_cfg() {
    return parse_toml(R"toml(
[LDM_SAMPLER]
sample_size = [256, 256]
step_size = 200
downscale = 8
latent_dims = 4
latent_scale = 0.18215
sampler_type = "ddim"
)toml");
}

// a fake sampler matching the adapter's expectations: init OK, run records
// the input and answers one tiny image
template <typename SAMPLER_INPUT, typename SAMPLER_OUTPUT>
class FakeSampler {
  public:
    SAMPLER_INPUT last_input{};
    int run_calls = 0;

    StatusCode init(const toml::table &) { return StatusCode::OK; }

    StatusCode run(const SAMPLER_INPUT &in, SAMPLER_OUTPUT &out) {
        last_input = in;
        ++run_calls;
        fill_output(out);
        return StatusCode::OK;
    }

  private:
    void fill_output(std_ddpm_output &out) const { out.out_images.push_back(cv::Mat(2, 2, CV_8UC3, cv::Scalar(1, 2, 3))); }
    void fill_output(std_ddim_output &out) const { out.sampled_images.push_back(cv::Mat(2, 2, CV_8UC3, cv::Scalar(1, 2, 3))); }
    void fill_output(std_cls_cond_ddim_output &out) const {
        out.sampled_images.push_back(cv::Mat(2, 2, CV_8UC3, cv::Scalar(1, 2, 3)));
    }
    void fill_output(std_ldm_output &out) const { out.sampled_image = cv::Mat(2, 2, CV_8UC3, cv::Scalar(1, 2, 3)); }
};

image_input make_request(const ParamSet *params) {
    image_input in;
    in.image.origin = jinq::models::io_define::common_io::byte_source::origin_kind::base64_text;
    in.image.data = "aGVsbG8=";  // payload ignored by generative models
    in.params = params;
    return in;
}

// expose the protected template seam for assertions
template <typename SAMPLER, typename SAMPLER_INPUT, typename SAMPLER_OUTPUT>
class TestAdapter : public DiffusionModelAdapter<SAMPLER, SAMPLER_INPUT, SAMPLER_OUTPUT> {
  public:
    using DiffusionModelAdapter<SAMPLER, SAMPLER_INPUT, SAMPLER_OUTPUT>::mutable_input;
    using DiffusionModelAdapter<SAMPLER, SAMPLER_INPUT, SAMPLER_OUTPUT>::sampler;
};

using Base64Output = jinq::models::io_define::common_io::base64_input;
using DdpmAdapter = TestAdapter<FakeSampler<std_ddpm_input, std_ddpm_output>, std_ddpm_input, std_ddpm_output>;
using DdimAdapter = TestAdapter<FakeSampler<std_ddim_input, std_ddim_output>, std_ddim_input, std_ddim_output>;
using ClsAdapter =
    TestAdapter<FakeSampler<std_cls_cond_ddim_input, std_cls_cond_ddim_output>, std_cls_cond_ddim_input, std_cls_cond_ddim_output>;
using LdmAdapter = TestAdapter<FakeSampler<std_ldm_input, std_ldm_output>, std_ldm_input, std_ldm_output>;

} // namespace

TEST(DiffusionParam, EmptyTomlFailsInit) {
    DdpmAdapter adapter;
    EXPECT_EQ(adapter.init(toml::table{}), StatusCode::MODEL_INIT_FAILED);
    EXPECT_FALSE(adapter.is_successfully_initialized());
}

TEST(DiffusionParam, MissingSampleSizeFailsInit) {
    DdpmAdapter adapter;
    auto cfg = parse_toml(R"toml(
[DDPM_SAMPLER]
timesteps = 1000
channels = 3
)toml");
    EXPECT_EQ(adapter.init(cfg), StatusCode::MODEL_INIT_FAILED);
}

TEST(DiffusionParam, ZeroSampleSizeFailsInit) {
    DdpmAdapter adapter;
    auto cfg = parse_toml(R"toml(
[DDPM_SAMPLER]
sample_size = [0, 128]
timesteps = 1000
channels = 3
)toml");
    EXPECT_EQ(adapter.init(cfg), StatusCode::MODEL_INIT_FAILED);
}

TEST(DiffusionParam, DdpmSeedsTemplateFromToml) {
    DdpmAdapter adapter;
    ASSERT_EQ(adapter.init(ddpm_cfg()), StatusCode::OK);
    EXPECT_EQ(adapter.mutable_input().sample_size, cv::Size(128, 128));
    EXPECT_EQ(adapter.mutable_input().timestep, 1000);
    EXPECT_EQ(adapter.mutable_input().channels, 3);
    EXPECT_FALSE(adapter.mutable_input().save_all_mid_results);

    Base64Output out;
    ASSERT_EQ(adapter.run(make_request(nullptr), out), StatusCode::OK);
    EXPECT_EQ(adapter.sampler().last_input.sample_size, cv::Size(128, 128));
    EXPECT_EQ(adapter.sampler().last_input.timestep, 1000);
    EXPECT_FALSE(adapter.sampler().last_input.save_all_mid_results);
}

TEST(DiffusionParam, DdpmTimestepsOverride) {
    DdpmAdapter adapter;
    ASSERT_EQ(adapter.init(ddpm_cfg()), StatusCode::OK);

    Base64Output out;
    ASSERT_EQ(adapter.run(make_request(nullptr), out), StatusCode::OK);
    EXPECT_EQ(adapter.sampler().last_input.timestep, 1000);  // template

    ParamSet params;
    params.set_i32("timesteps", 50);
    ASSERT_EQ(adapter.run(make_request(&params), out), StatusCode::OK);
    EXPECT_EQ(adapter.sampler().last_input.timestep, 50);
    EXPECT_EQ(adapter.mutable_input().timestep, 1000);  // template untouched
    EXPECT_EQ(adapter.sampler().last_input.sample_size, cv::Size(128, 128));
}

TEST(DiffusionParam, DdimStepsAndEtaOverride) {
    DdimAdapter adapter;
    ASSERT_EQ(adapter.init(ddim_cfg()), StatusCode::OK);
    EXPECT_EQ(adapter.mutable_input().sample_size, cv::Size(256, 256));
    EXPECT_EQ(adapter.mutable_input().total_steps, 1000);
    EXPECT_EQ(adapter.mutable_input().sample_steps, 100);

    Base64Output out;
    ParamSet params;
    params.set_i32("sample_steps", 20);
    params.set_f32("eta", 0.0f);
    ASSERT_EQ(adapter.run(make_request(&params), out), StatusCode::OK);
    EXPECT_EQ(adapter.sampler().last_input.sample_steps, 20);
    EXPECT_FLOAT_EQ(adapter.sampler().last_input.eta, 0.0f);
    EXPECT_EQ(adapter.sampler().last_input.total_steps, 1000);  // not overridable
    EXPECT_FALSE(adapter.sampler().last_input.save_all_mid_results);
}

TEST(DiffusionParam, ClsCondStepsEtaAndClassOverride) {
    ClsAdapter adapter;
    ASSERT_EQ(adapter.init(cls_cond_cfg()), StatusCode::OK);
    EXPECT_EQ(adapter.mutable_input().cls_id, 3);
    EXPECT_EQ(adapter.mutable_input().sample_size, cv::Size(128, 128));

    Base64Output out;
    ParamSet params;
    params.set_i32("sample_steps", 10);
    params.set_f32("eta", 1.0f);
    params.set_i32("cls_id", 42);
    ASSERT_EQ(adapter.run(make_request(&params), out), StatusCode::OK);
    EXPECT_EQ(adapter.sampler().last_input.sample_steps, 10);
    EXPECT_FLOAT_EQ(adapter.sampler().last_input.eta, 1.0f);
    EXPECT_EQ(adapter.sampler().last_input.cls_id, 42);
}

TEST(DiffusionParam, LdmStepSizeOverride) {
    LdmAdapter adapter;
    ASSERT_EQ(adapter.init(ldm_cfg()), StatusCode::OK);
    EXPECT_EQ(adapter.mutable_input().sample_size, cv::Size(256, 256));
    EXPECT_EQ(adapter.mutable_input().step_size, 200);

    Base64Output out;
    ParamSet params;
    params.set_i32("step_size", 25);
    ASSERT_EQ(adapter.run(make_request(&params), out), StatusCode::OK);
    EXPECT_EQ(adapter.sampler().last_input.step_size, 25);

    // unrelated keys are rejected by the envelope validator before reaching
    // here, so the adapter only ever sees whitelisted keys
    ASSERT_EQ(adapter.run(make_request(nullptr), out), StatusCode::OK);
    EXPECT_EQ(adapter.sampler().last_input.step_size, 200);
}

TEST(DiffusionParam, DdimMissingSampleStepsFailsInit) {
    DdimAdapter adapter;
    auto cfg = parse_toml(R"toml(
[DDIM_SAMPLER]
sample_size = [256, 256]
total_timesteps = 1000
channels = 3
)toml");
    EXPECT_EQ(adapter.init(cfg), StatusCode::MODEL_INIT_FAILED);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
