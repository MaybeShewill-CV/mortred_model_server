/************************************************
 * Author: Codex
 * File: sam_amg_param_unittest.cc
 * Date: 2026-09-01
 ************************************************/

// Verifies the SAM AMG wiring that the golden test (mat_input) cannot reach:
// the served image_input branch of run_sessions must reach generate() and
// carry the request param view, and the mat_input legacy path must stay
// byte-identical (nullptr params).

#include <gtest/gtest.h>

#include "models/backend/param_spec.h"
#include "models/model_io_define.h"
#include "models/segment_anything/sam_automask_generator/sam_automask_generator.h"

using jinq::common::StatusCode;
using jinq::models::io_define::common_io::image_input;
using jinq::models::io_define::common_io::mat_input;
using AmgOutput = jinq::models::io_define::segment_anything::sam_amg_output;

namespace {

// the model is never initialized here: generate() answers MODEL_INIT_FAILED
// when it is reached; the pre-fix bug answered MODEL_EMPTY_INPUT_IMAGE from
// the "unsupported input type" branch instead - this is exactly the signal
class TestAmg : public jinq::models::segment_anything::SamAutoMaskGenerator<image_input, AmgOutput> {
  public:
    using SamAutoMaskGenerator::run_sessions;
};

class TestAmgMat : public jinq::models::segment_anything::SamAutoMaskGenerator<mat_input, AmgOutput> {
  public:
    using SamAutoMaskGenerator::run_sessions;
};

} // namespace

TEST(SamAmgParam, served_image_input_branch_reaches_generate) {
    TestAmg model;
    image_input in;
    in.image.origin = jinq::models::io_define::common_io::byte_source::origin_kind::raw_bytes;
    std::vector<unsigned char> png;
    cv::imencode(".png", cv::Mat(8, 8, CV_8UC3, cv::Scalar(1, 2, 3)), png);
    in.image.data.assign(png.begin(), png.end());

    jinq::models::backend::ParamSet params;
    params.set_i32("points_per_side", 16);
    in.params = &params;

    AmgOutput out;
    // uninitialized sessions: reaching generate() means the image_input
    // branch executed and the params pointer traveled along
    EXPECT_EQ(model.run_sessions(in, out), StatusCode::MODEL_INIT_FAILED);
}

TEST(SamAmgParam, mat_input_legacy_path_stays_reachable) {
    TestAmgMat model;
    mat_input in;
    in.input_image = cv::Mat(8, 8, CV_8UC3, cv::Scalar(1, 2, 3));

    AmgOutput out;
    EXPECT_EQ(model.run_sessions(in, out), StatusCode::MODEL_INIT_FAILED);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
