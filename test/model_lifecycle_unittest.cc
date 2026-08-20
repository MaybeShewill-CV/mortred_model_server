/************************************************
 * Author: Codex
 * File: model_lifecycle_unittest.cc
 * Date: 2026-08-20
 *
 * P0 lifecycle contract tests for models:
 * 1. run() before init / after failed init must return MODEL_INIT_FAILED
 *    instead of dereferencing null backend tensors.
 * 2. Multi-backend init with missing/unknown BACKEND_DICT entries must fail
 *    cleanly instead of silently falling back to backend 0 (TRT) and
 *    dereferencing a missing section table.
 ************************************************/

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <toml/toml.hpp>

#include "common/status_code.h"
#include "models/classification/mobilenetv2.h"
#include "models/diffusion/autoencoder_kl.h"
#include "models/diffusion/cls_cond_ddpm_unet.h"
#include "models/diffusion/ddpm_unet.h"
#include "models/model_io_define.h"
#include "models/mono_depth_estimation/depth_anything.h"
#include "models/mono_depth_estimation/metric3d.h"
#include "models/object_detection/yolov8_detector.h"
#include "models/scene_segmentation/hrnet_segmentation.h"
#include "models/scene_segmentation/msocrnet.h"

namespace {

using jinq::common::StatusCode;
using jinq::models::io_define::classification::std_classification_output;
using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::diffusion::std_cls_cond_ddpm_unet_input;
using jinq::models::io_define::diffusion::std_cls_cond_ddpm_unet_output;
using jinq::models::io_define::diffusion::std_ddpm_unet_input;
using jinq::models::io_define::diffusion::std_ddpm_unet_output;
using jinq::models::io_define::diffusion::std_vae_decode_input;
using jinq::models::io_define::diffusion::std_vae_decode_output;
using jinq::models::io_define::mono_depth_estimation::std_mde_output;
using jinq::models::io_define::object_detection::std_object_detection_output;
using jinq::models::io_define::scene_segmentation::std_scene_segmentation_output;

cv::Mat make_test_image() {
    return cv::Mat(8, 8, CV_8UC3, cv::Scalar(30, 60, 90));
}

toml::table parse_toml_or_fail(const std::string& content) {
    auto parsed = toml::parse(content);
    if (!parsed) {
        ADD_FAILURE() << "fixture toml parse failed: "
                      << std::string(parsed.error().description());
        return toml::table{};
    }
    return std::move(parsed).table();
}

}  // namespace

/************************************************
 * P0-1: BaseAiModel::run must guard uninitialized models
 ************************************************/

TEST(ModelLifecycleGuard, RunBeforeInitReturnsInitFailed) {
    jinq::models::classification::MobileNetv2<mat_input, std_classification_output> model;
    mat_input in{make_test_image()};
    std_classification_output out;
    EXPECT_FALSE(model.is_successfully_initialized());
    EXPECT_EQ(model.run(in, out), StatusCode::MODEL_INIT_FAILED);
}

TEST(ModelLifecycleGuard, RunAfterFailedInitReturnsInitFailed) {
    jinq::models::classification::MobileNetv2<mat_input, std_classification_output> model;
    toml::table empty_cfg;
    EXPECT_EQ(model.init(empty_cfg), StatusCode::MODEL_INIT_FAILED);

    mat_input in{make_test_image()};
    std_classification_output out;
    EXPECT_EQ(model.run(in, out), StatusCode::MODEL_INIT_FAILED);
}

/************************************************
 * P0-3: multi-backend init must fail cleanly on bad configs
 ************************************************/

TEST(BackendConfigGuard, DdpmUnetEmptyConfigFailsCleanly) {
    jinq::models::diffusion::DDPMUNet<std_ddpm_unet_input, std_ddpm_unet_output> model;
    toml::table empty_cfg;
    EXPECT_EQ(model.init(empty_cfg), StatusCode::MODEL_INIT_FAILED);
}

TEST(BackendConfigGuard, DdpmUnetUnknownBackendNameFailsCleanly) {
    jinq::models::diffusion::DDPMUNet<std_ddpm_unet_input, std_ddpm_unet_output> model;
    auto cfg = parse_toml_or_fail(R"toml(
[DDPM_UNET]
[DDPM_UNET.backend]
type="nonexistent"
model_file_path="whatever.engine"
)toml");
    EXPECT_EQ(model.init(cfg), StatusCode::MODEL_INIT_FAILED);
}

TEST(BackendConfigGuard, ClsCondDdpmUnetEmptyConfigFailsCleanly) {
    jinq::models::diffusion::ClsCondDDPMUNet<std_cls_cond_ddpm_unet_input,
                                             std_cls_cond_ddpm_unet_output> model;
    toml::table empty_cfg;
    EXPECT_EQ(model.init(empty_cfg), StatusCode::MODEL_INIT_FAILED);
}

TEST(BackendConfigGuard, AutoencoderKlEmptyConfigFailsCleanly) {
    jinq::models::diffusion::AutoEncoderKL<std_vae_decode_input, std_vae_decode_output> model;
    toml::table empty_cfg;
    EXPECT_EQ(model.init(empty_cfg), StatusCode::MODEL_INIT_FAILED);
}

TEST(BackendConfigGuard, HrnetEmptyConfigFailsCleanly) {
    jinq::models::scene_segmentation::HRNetSegmentation<mat_input,
                                                        std_scene_segmentation_output> model;
    toml::table empty_cfg;
    EXPECT_EQ(model.init(empty_cfg), StatusCode::MODEL_INIT_FAILED);
}

TEST(BackendConfigGuard, Metric3dEmptyConfigFailsCleanly) {
    jinq::models::mono_depth_estimation::Metric3D<mat_input, std_mde_output> model;
    toml::table empty_cfg;
    EXPECT_EQ(model.init(empty_cfg), StatusCode::MODEL_INIT_FAILED);
}

TEST(BackendConfigGuard, DepthAnythingEmptyConfigFailsCleanly) {
    jinq::models::mono_depth_estimation::DepthAnything<mat_input, std_mde_output> model;
    toml::table empty_cfg;
    EXPECT_EQ(model.init(empty_cfg), StatusCode::MODEL_INIT_FAILED);
}

TEST(BackendConfigGuard, MsocrnetEmptyConfigFailsCleanly) {
    jinq::models::scene_segmentation::MsOcrNet<mat_input, std_scene_segmentation_output> model;
    toml::table empty_cfg;
    EXPECT_EQ(model.init(empty_cfg), StatusCode::MODEL_INIT_FAILED);
}

TEST(BackendConfigGuard, Yolov8UnknownBackendNameFailsCleanly) {
    jinq::models::object_detection::YoloV8Detector<mat_input, std_object_detection_output> model;
    auto cfg = parse_toml_or_fail(R"toml(
[YOLOV8]
[YOLOV8.backend]
type="nonexistent"
model_file_path="whatever.engine"
)toml");
    EXPECT_EQ(model.init(cfg), StatusCode::MODEL_INIT_FAILED);
}
