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
    std::string category;
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

// image classification
namespace classification {

struct cls_output {
    int class_id;
    std::string category;
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

enum DDPMSampler_Type {
    DDPM = 0,
    DDIM = 1,
};

struct ddpm_unet_input {
    std::vector<float> xt;
    int64_t timestep;
};
struct ddpm_unet_output {
    std::vector<float> predict_noise;
};
using std_ddpm_unet_input = ddpm_unet_input;
using std_ddpm_unet_output = ddpm_unet_output;

struct cls_cond_ddpm_unet_input {
    std::vector<float> xt;
    int64_t timestep;
    int cls_id = 0;
};
struct cls_cond_ddpm_unet_output {
    std::vector<float> predict_noise;
};
using std_cls_cond_ddpm_unet_input = cls_cond_ddpm_unet_input;
using std_cls_cond_ddpm_unet_output = cls_cond_ddpm_unet_output;

struct ddpm_sample_input {
    cv::Size sample_size;
    int timestep;
    int channels = 3;
    bool save_all_mid_results = true;
    bool use_fixed_noise_for_psample = false;
    bool save_raw_output = false;
};
struct ddpm_sample_output {
    std::vector<cv::Mat> out_images;
    std::vector<std::vector<float> > out_raw_predictions;
};
using std_ddpm_input = ddpm_sample_input;
using std_ddpm_output = ddpm_sample_output;

struct ddim_sample_input {
    cv::Size sample_size;
    int total_steps;
    int sample_steps;
    int channels = 3;
    bool save_all_mid_results = true;
    float* xt_data = nullptr;
    float eta = 1.0f;
    bool save_raw_output = false;
};
struct ddim_sample_output {
    std::vector<cv::Mat> sampled_images;
    std::vector<cv::Mat> predicted_x0;
    std::vector<std::vector<float> > raw_sampled_images;
    std::vector<std::vector<float> > raw_predicted_x0;
};
using std_ddim_input = ddim_sample_input;
using std_ddim_output = ddim_sample_output;

struct cls_cond_ddim_sample_input {
    cv::Size sample_size;
    int total_steps;
    int sample_steps;
    int cls_id= 0;
    int channels = 3;
    bool save_all_mid_results = true;
    float* xt_data = nullptr;
    float eta = 1.0f;
};
struct cls_cond_ddim_sample_output {
    std::vector<cv::Mat> sampled_images;
    std::vector<cv::Mat> predicted_x0;
};
using std_cls_cond_ddim_input = cls_cond_ddim_sample_input;
using std_cls_cond_ddim_output = cls_cond_ddim_sample_output;

struct autoencoder_kl_input {
    std::vector<float> decode_data;
};
struct autoencoder_kl_output {
    cv::Mat decode_output;
};
using std_vae_decode_input = autoencoder_kl_input;
using std_vae_decode_output = autoencoder_kl_output;

struct ldm_sample_input {
    cv::Size sample_size;
    int step_size;
    int downscale = 8;
    int latent_dims = 4;
    float latent_scale = 0.18215f;
    DDPMSampler_Type sampler_type = DDPMSampler_Type::DDIM;
};
struct ldm_sample_output {
    cv::Mat sampled_image;
};
using std_ldm_input = ldm_sample_input;
using std_ldm_output = ldm_sample_output;

}

// llm
namespace llm {

namespace text {

using token_id = int32_t;
using tokens = std::vector<int32_t >;

}

namespace embedding {

enum pool_type {
    EMBEDDING_MEAN_POOLING = 1,
    EMBEDDING_NONE_POOLING = 2,
};

struct embedding_input {
    std::string text;
    pool_type pooling_type = EMBEDDING_MEAN_POOLING;
};
struct embedding_output {
    std::vector<int32_t > token_ids;
    std::vector<std::vector<float> > token_embeds;
};

using std_embedding_input = embedding_input;
using std_embedding_output = embedding_output;

}

namespace vlm {

struct file_input {
    std::string image_path;
    std::string text;
};

struct mat_input {
    cv::Mat image;
    std::string text;
};

struct base64_input {
    std::string b64_image;
    std::string text;
};

struct bytes_input {
    unsigned char* image_bytes = nullptr;
    size_t bytes_length = 0;
    std::string text;
};

using std_vlm_input = file_input;
using std_vlm_output = std::string;

}

}

} // namespace io_define
} // namespace models
} // namespace jinq

#endif // MM_AI_SERVER_MODEL_IO_DEFINE_H
