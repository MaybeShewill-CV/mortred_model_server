/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: model_io_define.h
 * Date: 22-6-7
 ************************************************/

#ifndef MM_AI_SERVER_MODEL_IO_DEFINE_H
#define MM_AI_SERVER_MODEL_IO_DEFINE_H

#include <string>

#include <opencv2/opencv.hpp>

namespace jinq {
namespace models {
namespace io_define {

// common io
namespace common_io {

struct mat_input {
    cv::Mat input_image;
};

struct file_input {
    std::string input_image_path;
};

struct base64_input {
    std::string input_image_content;
};

struct pair_mat_input {
    cv::Mat src_input_image;
    cv::Mat dst_input_image;
};

} // namespace common_io

// image ocr
namespace ocr {

struct text_region {
    cv::Rect2f bbox;
    std::vector<cv::Point2f> polygon;
    float score;
};
using std_text_regions_output = std::vector<text_region>;

} // namespace ocr

// image object detection
namespace object_detection {

struct bbox {
    cv::Rect2f bbox;
    float score;
    int32_t class_id;
};
using std_object_detection_output = std::vector<bbox>;

struct face_bbox {
    cv::Rect2f bbox;
    float score;
    int32_t class_id;
    std::vector<cv::Point2f> landmarks;
};
using std_face_detection_output = std::vector<face_bbox>;

} // namespace object_detection

// image scene segmentation
namespace scene_segmentation {

struct seg_output {
    cv::Mat segmentation_result;
    cv::Mat colorized_seg_mask;
};
using std_scene_segmentation_output = seg_output;

} // namespace scene_segmentation

// image scene segmentation
namespace matting {

struct matting_output {
    cv::Mat matting_result;
};
using std_matting_output = matting_output;

} // namespace scene_segmentation

// image enhancement
namespace enhancement {

struct enhance_output {
    cv::Mat enhancement_result;
};
using std_enhancement_output = enhance_output;

} // namespace enhancement

// image enhancement
namespace classification {

struct cls_output {
    int class_id;
    std::vector<float> scores;
};
using std_classification_output = cls_output;

} // namespace classification

// image feature point
namespace feature_point {

struct fp {
    cv::Point2f location;
    std::vector<float> descriptor;
    float score;
};
using std_feature_point_output = std::vector<fp>;

struct matched_fp {
    std::pair<fp, fp> m_fp;
    float match_score;
};
using std_feature_point_match_output = std::vector<matched_fp>;

} // namespace feature_point

// mono depth estimation
namespace mono_depth_estimation {

struct mde_output {
    cv::Mat confidence_map;
    cv::Mat depth_map;
    cv::Mat colorized_depth_map;
};
using std_mde_output = mde_output;

} // namespace mono_depth_estimation

// diffusion
namespace diffusion {

struct ddpm_unet_input {
    std::vector<float> xt;
    int64_t timestep;
};

struct ddpm_unet_output {
    std::vector<float> predict_noise;
};
using std_ddpm_unet_input = ddpm_unet_input;
using std_ddpm_unet_output = ddpm_unet_output;

struct ddpm_sample_input {
    cv::Size sample_size;
    int64_t timestep;
    int channels = 3;
    bool save_all_mid_results = true;
};
struct ddpm_sample_output {
    std::vector<cv::Mat> out_images;
};
using std_ddpm_input = ddpm_sample_input;
using std_ddpm_output = ddpm_sample_output;

}

} // namespace io_define
} // namespace models
} // namespace jinq

#endif // MM_AI_SERVER_MODEL_IO_DEFINE_H
