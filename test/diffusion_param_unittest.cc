/************************************************
 * Author: Codex
 * File: diffusion_param_unittest.cc
 * Date: 2026-09-01
 ************************************************/

// Verifies the diffusion adapter's request-param plumbing without engines:
// fake samplers record the input they receive, so every override can be
// asserted against the config-template-plus-params resolution rule.

#include <gtest/gtest.h>

#include "factory/diffusion_model_adapter.h"
#include "models/backend/param_spec.h"

using jinq::common::StatusCode;
using jinq::factory::diffusion::DiffusionModelAdapter;
using jinq::models::backend::ParamSet;
using jinq::models::io_define::common_io::image_input;
using namespace jinq::models::io_define::diffusion;

namespace {

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

} // namespace

TEST(DiffusionParam, DdpmTimestepsOverride) {
    TestAdapter<FakeSampler<std_ddpm_input, std_ddpm_output>, std_ddpm_input, std_ddpm_output> adapter;
    ASSERT_EQ(adapter.init(toml::table{}), StatusCode::OK);
    adapter.mutable_input().timestep = 1000;

    Base64Output out;
    ASSERT_EQ(adapter.run(make_request(nullptr), out), StatusCode::OK);
    EXPECT_EQ(adapter.sampler().last_input.timestep, 1000);  // legacy: template

    ParamSet params;
    params.set_i32("timesteps", 50);
    ASSERT_EQ(adapter.run(make_request(&params), out), StatusCode::OK);
    EXPECT_EQ(adapter.sampler().last_input.timestep, 50);
    EXPECT_EQ(adapter.mutable_input().timestep, 1000);  // template untouched
}

TEST(DiffusionParam, DdimStepsAndEtaOverride) {
    TestAdapter<FakeSampler<std_ddim_input, std_ddim_output>, std_ddim_input, std_ddim_output> adapter;
    ASSERT_EQ(adapter.init(toml::table{}), StatusCode::OK);
    adapter.mutable_input().total_steps = 1000;
    adapter.mutable_input().sample_steps = 100;
    adapter.mutable_input().eta = 1.0f;

    Base64Output out;
    ParamSet params;
    params.set_i32("sample_steps", 20);
    params.set_f32("eta", 0.0f);
    ASSERT_EQ(adapter.run(make_request(&params), out), StatusCode::OK);
    EXPECT_EQ(adapter.sampler().last_input.sample_steps, 20);
    EXPECT_FLOAT_EQ(adapter.sampler().last_input.eta, 0.0f);
    EXPECT_EQ(adapter.sampler().last_input.total_steps, 1000);  // not overridable
}

TEST(DiffusionParam, ClsCondStepsEtaAndClassOverride) {
    TestAdapter<FakeSampler<std_cls_cond_ddim_input, std_cls_cond_ddim_output>, std_cls_cond_ddim_input,
                std_cls_cond_ddim_output>
        adapter;
    ASSERT_EQ(adapter.init(toml::table{}), StatusCode::OK);
    adapter.mutable_input().sample_steps = 100;
    adapter.mutable_input().eta = 0.5f;
    adapter.mutable_input().cls_id = 3;

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
    TestAdapter<FakeSampler<std_ldm_input, std_ldm_output>, std_ldm_input, std_ldm_output> adapter;
    ASSERT_EQ(adapter.init(toml::table{}), StatusCode::OK);
    adapter.mutable_input().step_size = 200;

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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
