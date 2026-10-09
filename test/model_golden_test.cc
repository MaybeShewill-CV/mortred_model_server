/************************************************
 * Author: Codex
 * File: model_golden_test.cc
 * Date: 2026-08-12
 *
 * 端到端模型 golden 测试（L2 级）。
 *
 * 标准用例只声明身份（用例名 / 配置 / 输入图 / creator / 输出类型），
 * 七步流程（权重检查 -> 配置加载与归一化 -> 建模型 -> init -> 读图 -> run
 * -> 与 test/golden/ 比对）全部由 model_golden_registry.h 提供。
 *
 * 以下用例刻意保持手写，不套宏：
 *   - batch 与单条一致性（3 个）：需要 run_batch 与逐条对照；
 *   - SAM prompt / AMG：输入是 prompt 结构或多 session；
 *   - CLIP：文本 + 图像双塔输入。
 *   - DDPM few-step：生成式工人是 image_input 适配器，不用读图宏。
 *
 * 行为约定：
 *   - 单条缺权重仍 GTEST_SKIP，原文仍是 weights not available；
 *     选中用例全部因此没有做数值比对时，进程退出码 77，
 *     标准输出恰好一行 NUMERICAL_NOT_RUN；
 *     CI 推理 job 设 MORTRED_CI_REQUIRE_WEIGHTS=1 时改为 FAIL，进程退出码 1；
 *   - MORTRED_UPDATE_GOLDEN=1 重新生成基线；
 *   - 配置里的 ../ 前缀被剥掉、backend 强制 cpu；
 *   - 用例名不变，--gtest_filter 行为不变。
 ************************************************/

#include "model_golden_registry.h"

#include <cstring>
#include <iostream>
#include <vector>

#include "factory/classification_task.h"
#include "factory/clip_task.h"
#include "factory/diffusion_model_adapter.h"
#include "factory/enhancement_task.h"
#include "factory/feature_embedding_task.h"
#include "factory/feature_point_task.h"
#include "factory/matting_task.h"
#include "factory/obj_detection_task.h"
#include "factory/ocr_task.h"
#include "factory/sam_task.h"
#include "factory/scene_segmentation_task.h"

#include "common/base64.h"
#include "models/backend/param_spec.h"
#include "models/diffusion/ddpm_sampler.h"
#include "models/io/diffusion.h"

// the registry owns the helpers and the IO type aliases; the hand-written
// cases below use them unqualified exactly as they did before the migration
using namespace jinq::test::golden;

// ============ golden 用例（保持迁移前的声明顺序） ============

GOLDEN_CLASSIFICATION_CASE(mobilenetv2_classification, "conf/model/classification/mobilenetv2/mobilenetv2_cpu.toml",
                           "demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG",
                           jinq::factory::classification::create_mobilenetv2_classifier, std_classification_output);

TEST(model_golden, mobilenetv2_batch_matches_single) {
    // batch=N must be numerically equivalent to N single runs (the batched
    // [N,H,W,3] session run is the whole point of the batching upgrade)
    std::string conf = "conf/model/classification/mobilenetv2/mobilenetv2_cpu.toml";
    if (!weights_available(conf))
        MORTRED_SKIP_OR_FAIL_WEIGHTS("weights not available");
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::classification::create_mobilenetv2_classifier<mat_input, std_classification_output>();
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG");
    ASSERT_FALSE(image.empty());

    std_classification_output single;
    ASSERT_EQ(model->run(mat_input{image}, single), StatusCode::OK);

    std::vector<mat_input> batch_inputs(4, mat_input{image});
    std::vector<std_classification_output> batch_outputs;
    std::vector<StatusCode> item_status;
    ASSERT_EQ(model->run_batch(batch_inputs, batch_outputs, item_status), StatusCode::OK);
    ASSERT_EQ(batch_outputs.size(), 4u);
    for (size_t idx = 0; idx < batch_outputs.size(); ++idx) {
        EXPECT_EQ(batch_outputs[idx].class_id, single.class_id) << "item " << idx;
        ASSERT_EQ(batch_outputs[idx].scores.size(), single.scores.size());
        for (size_t k = 0; k < single.scores.size(); ++k) {
            EXPECT_NEAR(batch_outputs[idx].scores[k], single.scores[k], k_score_tol) << "item " << idx << " score " << k;
        }
    }
}

TEST(model_golden, mobilenetv2_mixed_batch_isolates_failed_items) {
    // regression for the hand-written run_batch double misalignment: it split
    // the packed output by the TOTAL input count and indexed the pieces by the
    // ORIGINAL slot, silently cross-wiring results whenever one item failed
    // preprocessing (or failing the whole batch). The generic run_image_batch
    // must keep the valid items aligned with their single-run results.
    std::string conf = "conf/model/classification/mobilenetv2/mobilenetv2_cpu.toml";
    if (!weights_available(conf))
        MORTRED_SKIP_OR_FAIL_WEIGHTS("weights not available");
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::classification::create_mobilenetv2_classifier<mat_input, std_classification_output>();
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG");
    ASSERT_FALSE(image.empty());

    std_classification_output single;
    ASSERT_EQ(model->run(mat_input{image}, single), StatusCode::OK);

    // item 1 is an empty image: it must be isolated, its batch mates aligned
    std::vector<mat_input> batch_inputs;
    batch_inputs.push_back(mat_input{image});       // 0: valid
    batch_inputs.push_back(mat_input{cv::Mat()});   // 1: empty -> isolated
    batch_inputs.push_back(mat_input{image});       // 2: valid
    batch_inputs.push_back(mat_input{image});       // 3: valid
    std::vector<std_classification_output> batch_outputs;
    std::vector<StatusCode> item_status;
    ASSERT_EQ(model->run_batch(batch_inputs, batch_outputs, item_status), StatusCode::OK);
    ASSERT_EQ(batch_outputs.size(), 4u);
    ASSERT_EQ(item_status.size(), 4u);

    EXPECT_EQ(item_status[1], StatusCode::MODEL_EMPTY_INPUT_IMAGE) << "failed item must be isolated";
    for (const size_t idx : {0u, 2u, 3u}) {
        EXPECT_EQ(item_status[idx], StatusCode::OK) << "item " << idx;
        ASSERT_EQ(batch_outputs[idx].scores.size(), single.scores.size()) << "item " << idx;
        EXPECT_EQ(batch_outputs[idx].class_id, single.class_id) << "item " << idx;
        for (size_t k = 0; k < single.scores.size(); ++k) {
            EXPECT_NEAR(batch_outputs[idx].scores[k], single.scores[k], k_score_tol) << "item " << idx << " score " << k;
        }
    }
}

GOLDEN_CLASSIFICATION_CASE(resnet50_classification, "conf/model/classification/resnet/resnet_cpu.toml",
                           "demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG",
                           jinq::factory::classification::create_resnet_classifier, std_classification_output);

TEST(model_golden, densenet_batch_matches_single) {
    // generic smart-batch path (BackendCvModel::run_batch): packed [N,...]
    // run must be numerically equivalent to N single runs
    std::string conf = "conf/ci/densenet_onnx_hosted.toml";
    if (!weights_available(conf))
        MORTRED_SKIP_OR_FAIL_WEIGHTS("weights not available");
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::classification::create_densenet_classifier<mat_input, std_classification_output>();
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG");
    ASSERT_FALSE(image.empty());

    std_classification_output single;
    ASSERT_EQ(model->run(mat_input{image}, single), StatusCode::OK);

    std::vector<mat_input> batch_inputs(4, mat_input{image});
    std::vector<std_classification_output> batch_outputs;
    std::vector<StatusCode> item_status;
    ASSERT_EQ(model->run_batch(batch_inputs, batch_outputs, item_status), StatusCode::OK);
    ASSERT_EQ(batch_outputs.size(), 4u);
    for (size_t idx = 0; idx < batch_outputs.size(); ++idx) {
        EXPECT_EQ(item_status[idx], StatusCode::OK) << "item " << idx;
        EXPECT_EQ(batch_outputs[idx].class_id, single.class_id) << "item " << idx;
    }
}

GOLDEN_CLASSIFICATION_CASE(densenet121_classification, "conf/ci/densenet_onnx_hosted.toml",
                           "demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG",
                           jinq::factory::classification::create_densenet_classifier, std_classification_output);

GOLDEN_FEATURE_EMBEDDING_CASE(dinov2_feature_embedding, "conf/ci/dinov2_vitb14_onnx_hosted.toml",
                              "demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG",
                              jinq::factory::feature_embedding::create_dinov2_feature_extractor, std_feature_embedding_output);

GOLDEN_OBJECT_DETECTION_CASE(nanodet_detection, "conf/ci/nanodet_onnx_hosted.toml",
                             "demo_data/model_test_input/object_detection/bus.jpg",
                             jinq::factory::object_detection::create_nanodet_detector, std_object_detection_output);

GOLDEN_OBJECT_DETECTION_CASE(yolov5_detection, "conf/ci/yolov5_onnx_hosted.toml",
                             "demo_data/model_test_input/object_detection/bus.jpg", jinq::factory::object_detection::create_yolov5_detector,
                             std_object_detection_output);

GOLDEN_OBJECT_DETECTION_CASE(yolov6_detection, "conf/ci/yolov6_onnx_hosted.toml",
                             "demo_data/model_test_input/object_detection/bus.jpg", jinq::factory::object_detection::create_yolov6_detector,
                             std_object_detection_output);

GOLDEN_OBJECT_DETECTION_CASE(yolov7_detection, "conf/ci/yolov7_cpu_golden.toml",
                             "demo_data/model_test_input/object_detection/bus.jpg", jinq::factory::object_detection::create_yolov7_detector,
                             std_object_detection_output);

GOLDEN_OBJECT_DETECTION_CASE(yolov8_detection, "conf/ci/yolov8_onnx_hosted.toml",
                             "demo_data/model_test_input/object_detection/bus.jpg", jinq::factory::object_detection::create_yolov8_detector,
                             std_object_detection_output);

GOLDEN_OBJECT_DETECTION_CASE(yolov8_onnx_detection, "conf/ci/yolov8_onnx_hosted.toml",
                             "demo_data/model_test_input/object_detection/bus.jpg", jinq::factory::object_detection::create_yolov8_detector,
                             std_object_detection_output);

TEST(model_golden, yolov8_mixed_size_batch_matches_single_runs) {
    std::string conf = "conf/ci/yolov8_onnx_hosted.toml";
    if (!weights_available(conf))
        MORTRED_SKIP_OR_FAIL_WEIGHTS("weights not available");
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::object_detection::create_yolov8_detector<mat_input, std_object_detection_output>();
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);

    cv::Mat first = read_input_image("demo_data/model_test_input/object_detection/bus.jpg");
    cv::Mat second = read_input_image("demo_data/model_test_input/object_detection/horses.jpg");
    ASSERT_FALSE(first.empty());
    ASSERT_FALSE(second.empty());
    cv::resize(first, first, cv::Size(1032, 640));
    cv::resize(second, second, cv::Size(768, 512));

    std::vector<mat_input> inputs{mat_input{first}, mat_input{second}};
    std::vector<std_object_detection_output> singles;
    for (const auto &input : inputs) {
        std_object_detection_output output;
        ASSERT_EQ(model->run(input, output), StatusCode::OK);
        singles.push_back(std::move(output));
    }

    std::vector<std_object_detection_output> batch_outputs;
    std::vector<StatusCode> item_status;
    ASSERT_EQ(model->run_batch(inputs, batch_outputs, item_status), StatusCode::OK);
    ASSERT_EQ(item_status.size(), inputs.size());
    ASSERT_EQ(batch_outputs.size(), inputs.size());
    EXPECT_EQ(item_status[0], StatusCode::OK);
    EXPECT_EQ(item_status[1], StatusCode::OK);
    expect_equivalent_detections("yolov8 batch item 0", singles[0], batch_outputs[0]);
    expect_equivalent_detections("yolov8 batch item 1", singles[1], batch_outputs[1]);
}

GOLDEN_FACE_DETECTION_CASE(centerface_detection, "conf/ci/centerface_cpu_golden.toml",
                           "demo_data/model_test_input/object_detection/face_w_mask.jpg",
                           jinq::factory::object_detection::create_centerface_detector, std_face_detection_output);

GOLDEN_FACE_DETECTION_CASE(libface_detection, "conf/model/object_detection/libfacedetection/640x480_config.toml",
                           "demo_data/model_test_input/object_detection/face_wo_mask.jpg",
                           jinq::factory::object_detection::create_libface_detector, std_face_detection_output);

GOLDEN_SCENE_SEGMENTATION_CASE(bisenetv2_segmentation, "conf/ci/bisenetv2_onnx_hosted.toml",
                               "demo_data/model_test_input/scene_segmentation/cityscapes_test.png",
                               jinq::factory::scene_segmentation::create_bisenetv2_segmentor, std_scene_segmentation_output);

GOLDEN_SCENE_SEGMENTATION_CASE(pphuman_segmentation, "conf/ci/pphuman_onnx_hosted.toml",
                               "demo_data/model_test_input/scene_segmentation/human_image.jpg",
                               jinq::factory::scene_segmentation::create_pphuman_segmentor, std_scene_segmentation_output);

GOLDEN_TEXT_REGION_CASE(dbnet_text_detection, "conf/ci/dbnet_onnx_hosted.toml",
                        "demo_data/model_test_input/ocr/railway_ticket.png", jinq::factory::ocr::create_dbtext_detector,
                        std_text_regions_output);

GOLDEN_MATTING_CASE(modnet_matting, "conf/ci/modnet_onnx_hosted.toml", "demo_data/model_test_input/matting/matting_test.jpg",
                    jinq::factory::matting::create_modnet_segmentor, std_matting_output);

GOLDEN_MATTING_CASE(ppmatting_matting, "conf/ci/ppmatting_onnx_hosted.toml",
                    "demo_data/model_test_input/matting/matting_test.jpg", jinq::factory::matting::create_ppmatting_segmentor,
                    std_matting_output);

GOLDEN_ENHANCEMENT_CASE(enlightengan_enhancement, "conf/ci/enlightengan_onnx_hosted.toml",
                        "demo_data/model_test_input/enhancement/low_light/lol_test_1.png",
                        jinq::factory::enhancement::create_enlightengan_enhancementor, std_enhancement_output);

GOLDEN_ENHANCEMENT_CASE(attentivegan_enhancement, "conf/ci/attentive_gan_onnx_hosted.toml",
                        "demo_data/model_test_input/enhancement/derain/test_1.png",
                        jinq::factory::enhancement::create_attentivegan_enhancementor, std_enhancement_output);

GOLDEN_ENHANCEMENT_CASE(realesrgan_enhancement, "conf/ci/realesrgan_onnx_hosted.toml",
                        // colour input: a grayscale image cannot detect a BGR/RGB
                        // channel-order regression, because R == G == B there
                        "demo_data/model_test_input/enhancement/real_esr/test.JPG",
                        jinq::factory::enhancement::create_realesrgan_enhancementor, std_enhancement_output);

GOLDEN_KEYPOINT_CASE(superpoint_feature_point, "conf/ci/superpoint_onnx_hosted.toml",
                     "demo_data/model_test_input/feature_point/test.png", jinq::factory::feature_point::create_superpoint_extractor,
                     std_feature_point_output);

GOLDEN_RAW_MAT_CASE(fastsam_segmentation, "conf/model/segment_anything/fast_sam_s_config.toml", "demo_data/model_test_input/sam/truck.jpg",
                    jinq::factory::segment_anything::create_fast_sam_segmentor, cv::Mat);

TEST(model_golden, sam_prompt_prediction) {
    std::string conf = "conf/model/segment_anything/mobile_sam_config.toml";
    if (!weights_available(conf))
        MORTRED_SKIP_OR_FAIL_WEIGHTS("weights not available");
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::segment_anything::create_sam_predictor<jinq::models::io_define::segment_anything::sam_prompt_input,
                                                                       jinq::models::io_define::segment_anything::std_sam_prompt_output>();
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/sam/truck.jpg");
    ASSERT_FALSE(image.empty());
    jinq::models::io_define::segment_anything::sam_prompt_input input;
    input.image = image;
    input.bboxes = {cv::Rect(483, 683, 158, 132)};
    jinq::models::io_define::segment_anything::std_sam_prompt_output output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    ASSERT_FALSE(output.empty());
    expect_fingerprint("sam_prompt_prediction", output.front());
}

TEST(model_golden, sam_automask_generation) {
    std::string conf = "conf/model/segment_anything/mobile_sam_amg_config.toml";
    if (!weights_available(conf))
        MORTRED_SKIP_OR_FAIL_WEIGHTS("weights not available");
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::segment_anything::create_sam_auto_mask_generator<mat_input,
                                                                                 jinq::models::io_define::segment_anything::sam_amg_output>();
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/sam/truck.jpg");
    ASSERT_FALSE(image.empty());
    mat_input input{image};
    jinq::models::io_define::segment_anything::sam_amg_output output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    ASSERT_FALSE(output.segmentations.empty());
    expect_fingerprint("sam_automask_generation", output.segmentations.front());
}

TEST(model_golden, openai_clip_embedding) {
    std::string conf = "conf/model/openai_clip/vit_b_32_config.toml";
    if (!weights_available(conf))
        MORTRED_SKIP_OR_FAIL_WEIGHTS("weights not available");
    auto cfg = load_model_cfg(conf);
    auto model =
        jinq::factory::clip::create_openai_clip<jinq::models::io_define::clip::clip_input, jinq::models::io_define::clip::clip_output>();
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    jinq::models::io_define::clip::clip_input text_input;
    text_input.task_type = jinq::models::io_define::clip::ClipTaskType::TEXT_EMBEDDING;
    text_input.text = "a photo of fox";
    jinq::models::io_define::clip::clip_output text_output;
    ASSERT_EQ(model->run(text_input, text_output), StatusCode::OK);
    expect_embeddings("openai_clip_text_embedding", text_output.embeddings);

    cv::Mat image = read_input_image("demo_data/model_test_input/clip/fox.jpg");
    ASSERT_FALSE(image.empty());
    jinq::models::io_define::clip::clip_input input;
    input.task_type = jinq::models::io_define::clip::ClipTaskType::IMAGE_EMBEDDING;
    input.image = image;
    jinq::models::io_define::clip::clip_output output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    expect_embeddings("openai_clip_embedding", output.embeddings);
}

TEST(model_golden, ddpm_celeba_hq_fewstep) {
    // HTTP worker path: DiffusionModelAdapter + real UNet, not FakeSampler.
    // Few steps. Not in hosted (ONNX is ~143MiB). Nightly / local with weights.
    // Dummy images[] payload is ignored. Pixel fingerprint is not asserted:
    // p_sample still draws from std::random_device even when
    // use_fixed_noise_for_psample is set (that flag only reuses one noise
    // vector inside a single run).
    const std::string conf = "conf/ci/ddpm_onnx_fewstep.toml";
    if (!weights_available(conf))
        MORTRED_SKIP_OR_FAIL_WEIGHTS("weights not available");
    auto cfg = load_model_cfg(conf);
    using Sampler = jinq::models::diffusion::DDPMSampler<jinq::models::io_define::diffusion::std_ddpm_input,
                                                         jinq::models::io_define::diffusion::std_ddpm_output>;
    jinq::factory::diffusion::DiffusionModelAdapter<Sampler, jinq::models::io_define::diffusion::std_ddpm_input,
                                                    jinq::models::io_define::diffusion::std_ddpm_output>
        adapter;
    ASSERT_EQ(adapter.init(cfg), StatusCode::OK);

    jinq::models::backend::ParamSet params;
    params.set_i32("timesteps", 2);
    jinq::models::io_define::common_io::image_input request;
    request.image.origin = jinq::models::io_define::common_io::byte_source::origin_kind::base64_text;
    request.image.data = "aGVsbG8=";
    request.params = &params;

    jinq::models::io_define::common_io::base64_input output;
    ASSERT_EQ(adapter.run(request, output), StatusCode::OK);
    ASSERT_FALSE(output.input_image_content.empty());

    const std::string png_bytes = jinq::common::base64::decode(output.input_image_content);
    ASSERT_FALSE(png_bytes.empty());
    const std::vector<uchar> buffer(png_bytes.begin(), png_bytes.end());
    const cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
    ASSERT_FALSE(image.empty());
    EXPECT_EQ(image.cols, 128);
    EXPECT_EQ(image.rows, 128);
    EXPECT_EQ(image.channels(), 3);
}

#ifdef MORTRED_HAS_TRT
// Product yolov8s TRT path: load the product toml (path fix only, no force_cpu),
// run bus.jpg, and require every golden box with score >= 0.5 to match a unique
// prediction of the same class_id with IoU >= 0.5. Score near-equality is not
// required. Missing engine may SKIP; once the engine is present, FAIL on fail.
TEST(Yolov8TrtDetection, MatchesCpuGoldenBoxes) {
    const std::string conf = "conf/model/object_detection/yolov8/yolov8s.toml";
    if (!weights_available(conf)) {
        MORTRED_SKIP_OR_FAIL_WEIGHTS("tensorrt engine not available");
    }

    auto cfg_parsed = toml::parse_file(conf);
    ASSERT_TRUE(static_cast<bool>(cfg_parsed)) << "parse failed: " << conf;
    auto cfg = std::move(cfg_parsed).table();
    fix_toml_paths(cfg);

    auto model = jinq::factory::object_detection::create_yolov8_detector<mat_input, std_object_detection_output>();
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    const cv::Mat image = read_input_image("demo_data/model_test_input/object_detection/bus.jpg");
    ASSERT_FALSE(image.empty());

    std_object_detection_output output;
    ASSERT_EQ(model->run(mat_input{image}, output), StatusCode::OK);

    rapidjson::Document golden = load_golden_json("yolov8_detection");
    ASSERT_TRUE(golden.IsObject() && golden.HasMember("boxes") && golden["boxes"].IsArray());
    const auto &golden_boxes = golden["boxes"].GetArray();

    std::vector<bool> used(output.size(), false);
    for (rapidjson::SizeType gi = 0; gi < golden_boxes.Size(); ++gi) {
        const auto &g = golden_boxes[gi];
        if (g["score"].GetFloat() < 0.5f) {
            continue;
        }
        const int class_id = g["class_id"].GetInt();
        const cv::Rect2f gbox(g["x"].GetFloat(), g["y"].GetFloat(), g["w"].GetFloat(), g["h"].GetFloat());

        int best = -1;
        float best_iou = 0.0f;
        for (size_t pi = 0; pi < output.size(); ++pi) {
            if (used[pi] || output[pi].class_id != class_id) {
                continue;
            }
            const float iou = CvUtils::calc_iou(output[pi].bbox, gbox);
            if (iou > best_iou) {
                best_iou = iou;
                best = static_cast<int>(pi);
            }
        }
        ASSERT_GE(best, 0) << "no unused same-class prediction for golden box " << gi;
        EXPECT_GE(best_iou, k_box_iou_thresh) << "IoU below threshold for golden box " << gi;
        used[static_cast<size_t>(best)] = true;
    }
}
#endif // MORTRED_HAS_TRT

namespace {

bool skip_message_has(const testing::TestResult &result, const char *needle) {
    const int part_count = result.total_part_count();
    for (int i = 0; i < part_count; ++i) {
        const testing::TestPartResult &part = result.GetTestPartResult(i);
        if (!part.skipped()) {
            continue;
        }
        const char *message = part.message();
        if (message != nullptr && std::strstr(message, needle) != nullptr) {
            return true;
        }
    }
    return false;
}

// Called only after RUN_ALL_TESTS() returns, so gtest has already written
// --gtest_output xml. Exit 77 is the all-missing-weights case; it is not a
// failure and it is not a numerical pass.
int golden_process_exit(int gtest_result) {
    if (testing::GTEST_FLAG(list_tests)) {
        return gtest_result;
    }

    const testing::UnitTest *unit = testing::UnitTest::GetInstance();
    int selected = 0;
    int failed = 0;
    int weight_skips = 0;
    int numerical_done = 0;

    const int suite_count = unit->total_test_suite_count();
    for (int suite_index = 0; suite_index < suite_count; ++suite_index) {
        const testing::TestSuite *suite = unit->GetTestSuite(suite_index);
        const int test_count = suite->total_test_count();
        for (int test_index = 0; test_index < test_count; ++test_index) {
            const testing::TestInfo *info = suite->GetTestInfo(test_index);
            if (!info->should_run()) {
                continue;
            }
            ++selected;
            const testing::TestResult *result = info->result();
            if (result->Failed()) {
                ++failed;
                continue;
            }
            if (result->Skipped()) {
                if (skip_message_has(*result, "weights not available")) {
                    ++weight_skips;
                }
                continue;
            }
            if (result->Passed()) {
                ++numerical_done;
            }
        }
    }

    if (selected == 0) {
        return gtest_result;
    }
    if (failed > 0) {
        return 1;
    }
    if (weight_skips == selected) {
        std::cout << "NUMERICAL_NOT_RUN" << std::endl;
        return 77;
    }
    if (numerical_done == selected) {
        std::cout << "NUMERICAL_PASSED" << std::endl;
        return 0;
    }
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    const int gtest_result = RUN_ALL_TESTS();
    return golden_process_exit(gtest_result);
}
