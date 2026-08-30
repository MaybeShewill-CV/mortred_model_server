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
 *
 * 行为约定与迁移前完全一致：
 *   - 权重缺失时 GTEST_SKIP（本地 / 无 GPU 环境可跑）；
 *   - MORTRED_UPDATE_GOLDEN=1 重新生成基线；
 *   - 配置里的 ../ 前缀被剥掉、backend 强制 cpu；
 *   - 用例名不变，--gtest_filter 行为不变。
 ************************************************/

#include "model_golden_registry.h"

#include "factory/classification_task.h"
#include "factory/clip_task.h"
#include "factory/enhancement_task.h"
#include "factory/feature_point_task.h"
#include "factory/matting_task.h"
#include "factory/obj_detection_task.h"
#include "factory/ocr_task.h"
#include "factory/sam_task.h"
#include "factory/scene_segmentation_task.h"

// the registry owns the helpers and the IO type aliases; the hand-written
// cases below use them unqualified exactly as they did before the migration
using namespace jinq::test::golden;

// ============ golden 用例（保持迁移前的声明顺序） ============

GOLDEN_CLASSIFICATION_CASE(mobilenetv2_classification, "conf/model/classification/mobilenetv2/mobilenetv2_config.toml",
                           "demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG",
                           jinq::factory::classification::create_mobilenetv2_classifier, std_classification_output);

TEST(model_golden, mobilenetv2_batch_matches_single) {
    // batch=N must be numerically equivalent to N single runs (the batched
    // [N,H,W,3] session run is the whole point of the batching upgrade)
    std::string conf = "conf/model/classification/mobilenetv2/mobilenetv2_config.toml";
    if (!weights_available(conf))
        GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::classification::create_mobilenetv2_classifier<mat_input, std_classification_output>("mobilenetv2_golden");
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

GOLDEN_CLASSIFICATION_CASE(resnet50_classification, "conf/model/classification/resnet/resnet50_config.toml",
                           "demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG",
                           jinq::factory::classification::create_resnet_classifier, std_classification_output);

TEST(model_golden, densenet_batch_matches_single) {
    // generic smart-batch path (BackendCvModel::run_batch): packed [N,...]
    // run must be numerically equivalent to N single runs
    std::string conf = "conf/model/classification/densenet/densenet121_config.toml";
    if (!weights_available(conf))
        GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::classification::create_densenet_classifier<mat_input, std_classification_output>("densenet_golden");
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

GOLDEN_CLASSIFICATION_CASE(densenet121_classification, "conf/model/classification/densenet/densenet121_config.toml",
                           "demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG",
                           jinq::factory::classification::create_densenet_classifier, std_classification_output);

GOLDEN_CLASSIFICATION_CASE(dinov2_classification, "conf/model/classification/dinov2/dinov2_vitb14_config.toml",
                           "demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG",
                           jinq::factory::classification::create_dinov2_classifier, std_classification_output);

GOLDEN_OBJECT_DETECTION_CASE(nanodet_detection, "conf/model/object_detection/nano_det/nanodet_config.toml",
                             "demo_data/model_test_input/object_detection/bus.jpg",
                             jinq::factory::object_detection::create_nanodet_detector, std_object_detection_output);

GOLDEN_OBJECT_DETECTION_CASE(yolov5_detection, "conf/model/object_detection/yolov5/yolov5_config.toml",
                             "demo_data/model_test_input/object_detection/bus.jpg", jinq::factory::object_detection::create_yolov5_detector,
                             std_object_detection_output);

GOLDEN_OBJECT_DETECTION_CASE(yolov6_detection, "conf/model/object_detection/yolov6/yolov6_config.toml",
                             "demo_data/model_test_input/object_detection/bus.jpg", jinq::factory::object_detection::create_yolov6_detector,
                             std_object_detection_output);

GOLDEN_OBJECT_DETECTION_CASE(yolov7_detection, "conf/model/object_detection/yolov7/yolov7_config.toml",
                             "demo_data/model_test_input/object_detection/bus.jpg", jinq::factory::object_detection::create_yolov7_detector,
                             std_object_detection_output);

GOLDEN_OBJECT_DETECTION_CASE(yolov8_detection, "conf/model/object_detection/yolov8/yolov8_config.toml",
                             "demo_data/model_test_input/object_detection/bus.jpg", jinq::factory::object_detection::create_yolov8_detector,
                             std_object_detection_output);

TEST(model_golden, yolov8_mixed_size_batch_matches_single_runs) {
    std::string conf = "conf/model/object_detection/yolov8/yolov8_config.toml";
    if (!weights_available(conf))
        GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::object_detection::create_yolov8_detector<mat_input, std_object_detection_output>("yolov8_batch");
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

GOLDEN_FACE_DETECTION_CASE(centerface_detection, "conf/model/object_detection/centerface/centerface_config.toml",
                           "demo_data/model_test_input/object_detection/face_w_mask.jpg",
                           jinq::factory::object_detection::create_centerface_detector, std_face_detection_output);

GOLDEN_FACE_DETECTION_CASE(libface_detection, "conf/model/object_detection/libfacedetection/640x480_config.toml",
                           "demo_data/model_test_input/object_detection/face_wo_mask.jpg",
                           jinq::factory::object_detection::create_libface_detector, std_face_detection_output);

GOLDEN_SCENE_SEGMENTATION_CASE(bisenetv2_segmentation, "conf/model/scene_segmentation/bisenetv2/bisenetv2_config.toml",
                               "demo_data/model_test_input/scene_segmentation/cityscapes_test.png",
                               jinq::factory::scene_segmentation::create_bisenetv2_segmentor, std_scene_segmentation_output);

GOLDEN_SCENE_SEGMENTATION_CASE(pphuman_segmentation, "conf/model/scene_segmentation/pphuman/pphuman_config.toml",
                               "demo_data/model_test_input/scene_segmentation/human_image.jpg",
                               jinq::factory::scene_segmentation::create_pphuman_segmentor, std_scene_segmentation_output);

GOLDEN_TEXT_REGION_CASE(dbnet_text_detection, "conf/model/ocr/db_text_detector/dbnet_config.toml",
                        "demo_data/model_test_input/ocr/railway_ticket.png", jinq::factory::ocr::create_dbtext_detector,
                        std_text_regions_output);

GOLDEN_MATTING_CASE(modnet_matting, "conf/model/matting/modnet/modnet_config.toml", "demo_data/model_test_input/matting/matting_test.jpg",
                    jinq::factory::matting::create_modnet_segmentor, std_matting_output);

GOLDEN_MATTING_CASE(ppmatting_matting, "conf/model/matting/ppmatting/ppmatting_config.toml",
                    "demo_data/model_test_input/matting/matting_test.jpg", jinq::factory::matting::create_ppmatting_segmentor,
                    std_matting_output);

GOLDEN_ENHANCEMENT_CASE(enlightengan_enhancement, "conf/model/enhancement/enlighten_gan/enlightengan.toml",
                        "demo_data/model_test_input/enhancement/low_light/lol_test_1.png",
                        jinq::factory::enhancement::create_enlightengan_enhancementor, std_enhancement_output);

GOLDEN_ENHANCEMENT_CASE(attentivegan_enhancement, "conf/model/enhancement/attentive_gan_derain/attentive_gan_derain_config.toml",
                        "demo_data/model_test_input/enhancement/derain/test_1.png",
                        jinq::factory::enhancement::create_attentivegan_enhancementor, std_enhancement_output);

GOLDEN_ENHANCEMENT_CASE(realesrgan_enhancement, "conf/model/enhancement/real_esrgan/real_esrgan.toml",
                        "demo_data/model_test_input/enhancement/real_esr/wolf_gray.jpg",
                        jinq::factory::enhancement::create_realesrgan_enhancementor, std_enhancement_output);

GOLDEN_KEYPOINT_CASE(superpoint_feature_point, "conf/model/feature_point/superpoint/superpoint_config.toml",
                     "demo_data/model_test_input/feature_point/test.png", jinq::factory::feature_point::create_superpoint_extractor,
                     std_feature_point_output);

GOLDEN_RAW_MAT_CASE(fastsam_segmentation, "conf/model/segment_anything/fast_sam_s_config.toml", "demo_data/model_test_input/sam/truck.jpg",
                    jinq::factory::segment_anything::create_fast_sam_segmentor, cv::Mat);

TEST(model_golden, sam_prompt_prediction) {
    std::string conf = "conf/model/segment_anything/mobile_sam_config.toml";
    if (!weights_available(conf))
        GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::segment_anything::create_sam_predictor<jinq::models::io_define::segment_anything::sam_prompt_input,
                                                                       jinq::models::io_define::segment_anything::std_sam_prompt_output>(
        "sam_golden");
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
        GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::segment_anything::create_sam_auto_mask_generator<mat_input,
                                                                                 jinq::models::io_define::segment_anything::sam_amg_output>(
        "sam_amg_golden");
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
        GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model =
        jinq::factory::clip::create_openai_clip<jinq::models::io_define::clip::clip_input, jinq::models::io_define::clip::clip_output>(
            "openai_clip_golden");
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
