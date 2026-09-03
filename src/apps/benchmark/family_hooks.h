/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: family_hooks.h
 * Date: 26-9-3
 ************************************************/

#ifndef MORTRED_APPS_BENCHMARK_FAMILY_HOOKS_H
#define MORTRED_APPS_BENCHMARK_FAMILY_HOOKS_H

#include <algorithm>
#include <string>

#include "apps/benchmark/image_family.h"
#include "common/cv_utils.h"
#include "models/model_io_define.h"

namespace jinq {
namespace apps {
namespace benchmark {

using ClassificationOutput = jinq::models::io_define::classification::std_classification_output;
using ObjectOutput = jinq::models::io_define::object_detection::std_object_detection_output;
using FaceOutput = jinq::models::io_define::object_detection::std_face_detection_output;
using SegOutput = jinq::models::io_define::scene_segmentation::std_scene_segmentation_output;
using OcrOutput = jinq::models::io_define::ocr::std_text_regions_output;
using MattingOutput = jinq::models::io_define::matting::std_matting_output;
using EnhancementOutput = jinq::models::io_define::enhancement::std_enhancement_output;
using FeaturePointOutput = jinq::models::io_define::feature_point::std_feature_point_output;
using EmbeddingOutput = jinq::models::io_define::feature_embedding::std_feature_embedding_output;
using DepthOutput = jinq::models::io_define::mono_depth_estimation::std_mde_output;
using AmgOutput = jinq::models::io_define::segment_anything::std_sam_amg_output;

inline ImageFamilyHooks<ClassificationOutput> classification_hooks() {
    ImageFamilyHooks<ClassificationOutput> hooks;
    hooks.default_image = "../demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG";
    hooks.output_dir = "../demo_data/model_test_input/classification";
    hooks.loops = 1000;
    hooks.handle_output = [](const cv::Mat &, const ClassificationOutput &out, const std::string &,
                             const std::string &) {
        LOG(INFO) << "classify id: " << out.class_id;
        if (out.scores.empty()) {
            return;
        }
        const auto max_score = std::max_element(out.scores.begin(), out.scores.end());
        LOG(INFO) << "max classify score: " << *max_score;
        LOG(INFO) << "max classify id: " << static_cast<int>(std::distance(out.scores.begin(), max_score));
    };
    return hooks;
}

inline ImageFamilyHooks<ObjectOutput> object_detection_hooks() {
    ImageFamilyHooks<ObjectOutput> hooks;
    hooks.default_image = "../demo_data/model_test_input/object_detection/horses.jpg";
    hooks.output_dir = "../demo_data/model_test_input/object_detection";
    hooks.handle_output = [](const cv::Mat &src, const ObjectOutput &out, const std::string &image_path,
                             const std::string &model_id) {
        cv::Mat vis = src.clone();
        jinq::common::CvUtils::vis_object_detection(vis, out, 80);
        const std::string output_path = result_image_path(
            "../demo_data/model_test_input/object_detection", image_path, model_id);
        cv::imwrite(output_path, vis);
        LOG(INFO) << "detection result image has been written into: " << output_path;
    };
    return hooks;
}

inline ImageFamilyHooks<FaceOutput> face_detection_hooks() {
    ImageFamilyHooks<FaceOutput> hooks;
    hooks.default_image = "../demo_data/model_test_input/object_detection/face_wo_mask.jpg";
    hooks.output_dir = "../demo_data/model_test_input/object_detection";
    hooks.handle_output = [](const cv::Mat &src, const FaceOutput &out, const std::string &image_path,
                             const std::string &model_id) {
        cv::Mat vis = src.clone();
        jinq::common::CvUtils::vis_object_detection(vis, out, 80);
        const std::string output_path = result_image_path(
            "../demo_data/model_test_input/object_detection", image_path, model_id);
        cv::imwrite(output_path, vis);
        LOG(INFO) << "detection result image has been written into: " << output_path;
    };
    return hooks;
}

inline ImageFamilyHooks<SegOutput> scene_segmentation_hooks() {
    ImageFamilyHooks<SegOutput> hooks;
    hooks.default_image = "../demo_data/model_test_input/scene_segmentation/cityscapes_test.png";
    hooks.output_dir = "../demo_data/model_test_input/scene_segmentation";
    hooks.handle_output = [](const cv::Mat &, const SegOutput &out, const std::string &image_path,
                             const std::string &model_id) {
        cv::Mat color;
        jinq::common::CvUtils::colorize_segmentation_mask(out.segmentation_result, color, 80);
        const std::string output_path = result_image_path(
            "../demo_data/model_test_input/scene_segmentation", image_path, model_id);
        cv::imwrite(output_path, color);
        LOG(INFO) << "segmentation result image has been written into: " << output_path;
    };
    return hooks;
}

inline ImageFamilyHooks<OcrOutput> ocr_hooks() {
    ImageFamilyHooks<OcrOutput> hooks;
    hooks.default_image = "../demo_data/model_test_input/ocr/railway_ticket.png";
    hooks.output_dir = "../demo_data/model_test_input/ocr";
    hooks.handle_output = [](const cv::Mat &src, const OcrOutput &out, const std::string &image_path,
                             const std::string &model_id) {
        cv::Mat vis = src.clone();
        jinq::common::CvUtils::vis_text_detection(vis, out);
        const std::string output_path = result_image_path("../demo_data/model_test_input/ocr", image_path, model_id);
        cv::imwrite(output_path, vis);
        LOG(INFO) << "detection result image has been written into: " << output_path;
    };
    return hooks;
}

inline ImageFamilyHooks<MattingOutput> matting_hooks() {
    ImageFamilyHooks<MattingOutput> hooks;
    hooks.default_image = "../demo_data/model_test_input/matting/matting_test.jpg";
    hooks.output_dir = "../demo_data/model_test_input/matting";
    hooks.handle_output = [](const cv::Mat &, const MattingOutput &out, const std::string &image_path,
                             const std::string &model_id) {
        const std::string output_path =
            result_image_path("../demo_data/model_test_input/matting", image_path, model_id);
        cv::imwrite(output_path, out.matting_result);
        LOG(INFO) << "matting result image has been written into: " << output_path;
    };
    return hooks;
}

inline ImageFamilyHooks<EnhancementOutput> enhancement_hooks() {
    ImageFamilyHooks<EnhancementOutput> hooks;
    hooks.default_image = "../demo_data/model_test_input/enhancement/derain/test_1.png";
    hooks.output_dir = "../demo_data/model_test_input/enhancement";
    hooks.handle_output = [](const cv::Mat &, const EnhancementOutput &out, const std::string &image_path,
                             const std::string &model_id) {
        const std::string output_path =
            result_image_path("../demo_data/model_test_input/enhancement", image_path, model_id);
        cv::imwrite(output_path, out.enhancement_result);
        LOG(INFO) << "enhancement result image has been written into: " << output_path;
    };
    return hooks;
}

inline ImageFamilyHooks<FeaturePointOutput> feature_point_hooks() {
    ImageFamilyHooks<FeaturePointOutput> hooks;
    hooks.default_image = "../demo_data/model_test_input/feature_point/test.png";
    hooks.output_dir = "../demo_data/model_test_input/feature_point";
    hooks.handle_output = [](const cv::Mat &src, const FeaturePointOutput &out, const std::string &image_path,
                             const std::string &model_id) {
        cv::Mat vis = src.clone();
        jinq::common::CvUtils::vis_feature_points(vis, out, 4);
        const std::string output_path =
            result_image_path("../demo_data/model_test_input/feature_point", image_path, model_id);
        cv::imwrite(output_path, vis);
        LOG(INFO) << "feature point extract result image has been written into: " << output_path;
    };
    return hooks;
}

inline ImageFamilyHooks<EmbeddingOutput> feature_embedding_hooks() {
    ImageFamilyHooks<EmbeddingOutput> hooks;
    hooks.default_image = "../demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG";
    hooks.output_dir = "../demo_data/model_test_input/classification";
    hooks.handle_output = [](const cv::Mat &, const EmbeddingOutput &out, const std::string &, const std::string &) {
        LOG(INFO) << "image embedding features dims: " << out.embedding.size();
        if (out.embedding.size() >= 5) {
            LOG(INFO) << "image embedding features: " << out.embedding[0] << " " << out.embedding[1] << " "
                      << out.embedding[2] << " " << out.embedding[3] << " " << out.embedding[4];
        }
    };
    return hooks;
}

inline ImageFamilyHooks<DepthOutput> mono_depth_hooks() {
    ImageFamilyHooks<DepthOutput> hooks;
    hooks.default_image = "../demo_data/model_test_input/mono_depth_estimation/0000000005.png";
    hooks.output_dir = "../demo_data/model_test_input/mono_depth_estimation";
    hooks.loops = 10;
    hooks.handle_output = [](const cv::Mat &, const DepthOutput &out, const std::string &image_path,
                             const std::string &model_id) {
        const std::string output_path = result_image_path(
            "../demo_data/model_test_input/mono_depth_estimation", image_path, model_id);
        cv::imwrite(output_path, out.colorized_depth_map);
        LOG(INFO) << "prediction colorized depth image has been written into: " << output_path;
    };
    return hooks;
}

inline ImageFamilyHooks<AmgOutput> sam_amg_hooks() {
    ImageFamilyHooks<AmgOutput> hooks;
    hooks.default_image = "../demo_data/model_test_input/sam/truck.jpg";
    hooks.output_dir = "../demo_data/model_test_input/sam";
    hooks.loops = 10;
    hooks.warmup = false;
    hooks.handle_output = [](const cv::Mat &src, const AmgOutput &out, const std::string &image_path,
                             const std::string &model_id) {
        if (out.segmentations.empty()) {
            LOG(WARNING) << "SAM AMG produced no masks";
            return;
        }
        const auto counts = static_cast<int>(out.segmentations.size());
        auto color_pool = jinq::common::CvUtils::generate_color_map(counts + 1);
        cv::Mat color_mask = cv::Mat::zeros(out.segmentations[0].size(), CV_8UC3);
        for (size_t idx = 0; idx < out.segmentations.size(); ++idx) {
            const auto &mask = out.segmentations[idx];
            const auto color = color_pool[static_cast<int>(idx)];
            for (int row = 0; row < mask.rows; ++row) {
                const auto mask_row = mask.ptr<float>(row);
                auto color_row = color_mask.ptr<cv::Vec3b>(row);
                for (int col = 0; col < mask.cols; ++col) {
                    if (mask_row[col] == 255.0f) {
                        color_row[col][0] = static_cast<uchar>(color[0]);
                        color_row[col][1] = static_cast<uchar>(color[1]);
                        color_row[col][2] = static_cast<uchar>(color[2]);
                    }
                }
            }
            if (idx < out.bboxes.size()) {
                cv::rectangle(color_mask, out.bboxes[idx], color, 2);
            }
        }
        cv::Mat merged;
        cv::addWeighted(src, 0.6, color_mask, 0.4, 0.0, merged);
        const std::string output_path =
            result_image_path("../demo_data/model_test_input/sam", image_path, model_id);
        cv::imwrite(output_path, merged);
        LOG(INFO) << "sam amg prediction result image has been written into: " << output_path;
    };
    return hooks;
}

} // namespace benchmark
} // namespace apps
} // namespace jinq

#endif // MORTRED_APPS_BENCHMARK_FAMILY_HOOKS_H
