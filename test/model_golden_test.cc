/************************************************
 * Author: Codex
 * File: model_golden_test.cc
 * Date: 2026-08-12
 *
 * 视觉模型推理黄金回归测试（L2）：
 * - 固定输入 + 固定配置运行真实模型，与 test/golden/ 中的基线输出比较，
 *   用于快速验证"修改模型推理函数后行为未改变"。
 * - 权重不在 git 仓库：权重缺失时 GTEST_SKIP（本地/GPU 机器才执行）。
 * - MORTRED_UPDATE_GOLDEN=1 时重新生成黄金文件（不比较）。
 * - 配置统一处理：相对路径修正（../xxx -> xxx）、compute_backend 强制 cpu，
 *   保证结果可移植与确定性。
 * - 覆盖模型按难度递增：分类 -> 检测 -> 分割 -> OCR -> Matting -> 增强 ->
 *   特征点 -> SAM -> CLIP。
 ************************************************/

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "common/file_path_util.h"
#include "common/status_code.h"
#include "factory/classification_task.h"
#include "factory/clip_task.h"
#include "factory/enhancement_task.h"
#include "factory/feature_point_task.h"
#include "factory/matting_task.h"
#include "factory/obj_detection_task.h"
#include "factory/ocr_task.h"
#include "factory/sam_task.h"
#include "factory/scene_segmentation_task.h"
#include "models/model_io_define.h"

using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::models::io_define::classification::std_classification_output;
using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::enhancement::std_enhancement_output;
using jinq::models::io_define::feature_point::std_feature_point_output;
using jinq::models::io_define::matting::std_matting_output;
using jinq::models::io_define::object_detection::bbox;
using jinq::models::io_define::object_detection::face_bbox;
using jinq::models::io_define::object_detection::std_face_detection_output;
using jinq::models::io_define::object_detection::std_object_detection_output;
using jinq::models::io_define::ocr::std_text_regions_output;
using jinq::models::io_define::scene_segmentation::std_scene_segmentation_output;

namespace {

constexpr double kScoreTol = 1e-3;
constexpr double kDetScoreTol = 1e-2;
constexpr float kBoxIouThresh = 0.5f;
constexpr double kFingerprintDiff = 1.0;
constexpr double kKeypointMatchDist = 3.0;
constexpr double kEmbeddingCosThresh = 0.999;

bool update_golden_mode() {
    const char* env = std::getenv("MORTRED_UPDATE_GOLDEN");
    return env != nullptr && std::string(env) == "1";
}

std::string golden_path(const std::string& name, const std::string& ext) {
    return "test/golden/" + name + ext;
}

/*** 配置：解析 conf 文件，修正相对路径，强制 cpu 后端 */
void fix_toml_paths(toml::value& value) {
    if (value.is_table()) {
        for (auto& item : value.as_table()) {
            if (item.second.is_string()) {
                std::string s = item.second.as_string();
                if (s.rfind("../", 0) == 0) {
                    item.second = s.substr(3);
                }
            } else {
                fix_toml_paths(item.second);
            }
        }
    } else if (value.is_array()) {
        for (auto& item : value.as_array()) {
            fix_toml_paths(item);
        }
    }
}

void force_cpu_backend(toml::value& value) {
    if (value.is_table()) {
        for (auto& item : value.as_table()) {
            if (item.first == "compute_backend") {
                item.second = std::string("cpu");
            } else {
                force_cpu_backend(item.second);
            }
        }
    } else if (value.is_array()) {
        for (auto& item : value.as_array()) {
            force_cpu_backend(item);
        }
    }
}

toml::value load_model_cfg(const std::string& conf_rel_path) {
    auto cfg = toml::parse(conf_rel_path);
    fix_toml_paths(cfg);
    force_cpu_backend(cfg);
    return cfg;
}

cv::Mat read_input_image(const std::string& path) {
    return cv::imread(path, cv::IMREAD_COLOR);
}

/*** 黄金 JSON 读写 */
std::string serialize_json(const rapidjson::Document& doc) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    return buffer.GetString();
}

rapidjson::Document load_golden_json(const std::string& name) {
    rapidjson::Document doc;
    std::ifstream in(golden_path(name, ".json"));
    if (in.is_open()) {
        std::stringstream ss;
        ss << in.rdbuf();
        doc.Parse(ss.str().c_str());
    }
    return doc;
}

void write_golden_text(const std::string& name, const std::string& ext, const std::string& content) {
    std::string path = golden_path(name, ext);
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path);
    out << content;
}

/*** 图像指纹：任意类型 Mat -> 64x64 CV_8UC3，用于可移植比较 */
cv::Mat make_fingerprint(const cv::Mat& src) {
    cv::Mat normalized;
    if (src.type() == CV_32SC1) {
        double mn = 0.0, mx = 0.0;
        cv::minMaxIdx(src, &mn, &mx);
        double scale = mx > mn ? 255.0 / (mx - mn) : 1.0;
        src.convertTo(normalized, CV_8UC1, scale, -mn * scale);
    } else if (src.type() == CV_32FC1 || src.type() == CV_32FC3) {
        cv::Mat tmp;
        cv::normalize(src, tmp, 0, 255, cv::NORM_MINMAX);
        tmp.convertTo(normalized, src.channels() == 1 ? CV_8UC1 : CV_8UC3);
    } else {
        normalized = src.clone();
    }
    cv::Mat resized;
    cv::resize(normalized, resized, cv::Size(64, 64), 0, 0, cv::INTER_AREA);
    if (resized.channels() == 1) {
        cv::cvtColor(resized, resized, cv::COLOR_GRAY2BGR);
    }
    return resized;
}

void save_golden_fingerprint(const std::string& name, const cv::Mat& mat) {
    std::string path = golden_path(name, ".png");
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    cv::imwrite(path, make_fingerprint(mat));
}

void expect_fingerprint(const std::string& name, const cv::Mat& mat) {
    std::string path = golden_path(name, ".png");
    if (update_golden_mode()) {
        save_golden_fingerprint(name, mat);
        GTEST_SKIP() << "golden updated: " << path;
    }
    cv::Mat golden = cv::imread(path, cv::IMREAD_COLOR);
    ASSERT_FALSE(golden.empty()) << "golden missing, run with MORTRED_UPDATE_GOLDEN=1: " << path;
    cv::Mat current = make_fingerprint(mat);
    ASSERT_EQ(golden.size(), current.size());
    cv::Mat diff;
    cv::absdiff(golden, current, diff);
    double mean = cv::mean(diff)[0];
    EXPECT_LE(mean, kFingerprintDiff) << "fingerprint drift for " << name << ", mean abs diff = " << mean;
}

void expect_scores(const std::string& name, const std_classification_output& output) {
    rapidjson::Document golden = load_golden_json(name);
    if (update_golden_mode()) {
        rapidjson::Document doc;
        doc.SetObject();
        rapidjson::Document::AllocatorType& a = doc.GetAllocator();
        doc.AddMember("class_id", output.class_id, a);
        rapidjson::Value scores(rapidjson::kArrayType);
        for (float s : output.scores) scores.PushBack(s, a);
        doc.AddMember("scores", scores, a);
        write_golden_text(name, ".json", serialize_json(doc));
        GTEST_SKIP() << "golden updated: " << name;
    }
    ASSERT_TRUE(golden.IsObject() && golden.HasMember("class_id") && golden.HasMember("scores"))
        << "golden missing, run with MORTRED_UPDATE_GOLDEN=1: " << name;
    EXPECT_EQ(output.class_id, golden["class_id"].GetInt());
    ASSERT_EQ(output.scores.size(), golden["scores"].GetArray().Size());
    size_t idx = 0;
    for (const auto& s : golden["scores"].GetArray()) {
        EXPECT_NEAR(output.scores[idx], s.GetFloat(), kScoreTol) << "score mismatch at " << idx;
        ++idx;
    }
}

float calc_iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);
    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float inter = w * h;
    float uni = a.width * a.height + b.width * b.height - inter;
    return uni > 0 ? inter / uni : 0.0f;
}

const std::vector<cv::Point2f>& get_landmarks(const face_bbox& box) {
    return box.landmarks;
}

const std::vector<cv::Point2f>& get_landmarks(const bbox&) {
    static const std::vector<cv::Point2f> kEmpty;
    return kEmpty;
}

template <typename BoxT>
rapidjson::Value serialize_box(const BoxT& box, rapidjson::Document::AllocatorType& a) {
    rapidjson::Value obj(rapidjson::kObjectType);
    obj.AddMember("x", box.bbox.x, a);
    obj.AddMember("y", box.bbox.y, a);
    obj.AddMember("w", box.bbox.width, a);
    obj.AddMember("h", box.bbox.height, a);
    obj.AddMember("score", box.score, a);
    obj.AddMember("class_id", box.class_id, a);
    return obj;
}

template <typename BoxT>
void expect_boxes(const std::string& name, const std::vector<BoxT>& boxes, bool has_landmarks) {
    rapidjson::Document golden = load_golden_json(name);
    if (update_golden_mode()) {
        rapidjson::Document doc;
        doc.SetObject();
        rapidjson::Document::AllocatorType& a = doc.GetAllocator();
        rapidjson::Value arr(rapidjson::kArrayType);
        for (const auto& box : boxes) {
            rapidjson::Value obj = serialize_box(box, a);
            if (has_landmarks) {
                rapidjson::Value pts(rapidjson::kArrayType);
                for (const auto& p : get_landmarks(box)) {
                    rapidjson::Value pt(rapidjson::kArrayType);
                    pt.PushBack(p.x, a);
                    pt.PushBack(p.y, a);
                    pts.PushBack(pt, a);
                }
                obj.AddMember("landmarks", pts, a);
            }
            arr.PushBack(obj, a);
        }
        doc.AddMember("boxes", arr, a);
        write_golden_text(name, ".json", serialize_json(doc));
        GTEST_SKIP() << "golden updated: " << name;
    }
    ASSERT_TRUE(golden.IsObject() && golden.HasMember("boxes") && golden["boxes"].IsArray())
        << "golden missing, run with MORTRED_UPDATE_GOLDEN=1: " << name;

    const auto& golden_boxes = golden["boxes"].GetArray();
    ASSERT_EQ(boxes.size(), golden_boxes.Size()) << "detection count changed for " << name;

    std::vector<bool> matched(golden_boxes.Size(), false);
    for (const auto& box : boxes) {
        int best = -1;
        float best_iou = 0.0f;
        for (rapidjson::SizeType i = 0; i < golden_boxes.Size(); ++i) {
            if (matched[i]) continue;
            const auto& g = golden_boxes[i];
            cv::Rect2f gbox(g["x"].GetFloat(), g["y"].GetFloat(), g["w"].GetFloat(), g["h"].GetFloat());
            float iou = calc_iou(box.bbox, gbox);
            if (iou > best_iou) {
                best_iou = iou;
                best = static_cast<int>(i);
            }
        }
        ASSERT_GE(best, 0) << "unmatched detection for " << name;
        EXPECT_GE(best_iou, kBoxIouThresh) << "low IoU for " << name;
        const auto& g = golden_boxes[best];
        EXPECT_NEAR(box.score, g["score"].GetFloat(), kDetScoreTol);
        EXPECT_EQ(box.class_id, g["class_id"].GetInt());
        if (has_landmarks && g.HasMember("landmarks")) {
            ASSERT_EQ(get_landmarks(box).size(), g["landmarks"].GetArray().Size());
            rapidjson::SizeType li = 0;
            for (const auto& lp : g["landmarks"].GetArray()) {
                cv::Point2f gp(lp[0].GetFloat(), lp[1].GetFloat());
                EXPECT_LE(cv::norm(get_landmarks(box)[li] - gp), kKeypointMatchDist);
                ++li;
            }
        }
        matched[best] = true;
    }
}

void expect_text_regions(const std::string& name, const std_text_regions_output& regions) {
    rapidjson::Document golden = load_golden_json(name);
    if (update_golden_mode()) {
        rapidjson::Document doc;
        doc.SetObject();
        rapidjson::Document::AllocatorType& a = doc.GetAllocator();
        rapidjson::Value arr(rapidjson::kArrayType);
        for (const auto& region : regions) {
            rapidjson::Value obj(rapidjson::kObjectType);
            obj.AddMember("x", region.bbox.x, a);
            obj.AddMember("y", region.bbox.y, a);
            obj.AddMember("w", region.bbox.width, a);
            obj.AddMember("h", region.bbox.height, a);
            obj.AddMember("score", region.score, a);
            arr.PushBack(obj, a);
        }
        doc.AddMember("regions", arr, a);
        write_golden_text(name, ".json", serialize_json(doc));
        GTEST_SKIP() << "golden updated: " << name;
    }
    ASSERT_TRUE(golden.IsObject() && golden.HasMember("regions") && golden["regions"].IsArray())
        << "golden missing, run with MORTRED_UPDATE_GOLDEN=1: " << name;

    const auto& golden_regions = golden["regions"].GetArray();
    ASSERT_EQ(regions.size(), golden_regions.Size()) << "text region count changed for " << name;
    std::vector<bool> matched(golden_regions.Size(), false);
    for (const auto& region : regions) {
        int best = -1;
        float best_iou = 0.0f;
        for (rapidjson::SizeType i = 0; i < golden_regions.Size(); ++i) {
            if (matched[i]) continue;
            const auto& g = golden_regions[i];
            cv::Rect2f gbox(g["x"].GetFloat(), g["y"].GetFloat(), g["w"].GetFloat(), g["h"].GetFloat());
            float iou = calc_iou(region.bbox, gbox);
            if (iou > best_iou) {
                best_iou = iou;
                best = static_cast<int>(i);
            }
        }
        ASSERT_GE(best, 0) << "unmatched text region for " << name;
        EXPECT_GE(best_iou, kBoxIouThresh) << "low IoU for " << name;
        EXPECT_NEAR(region.score, golden_regions[best]["score"].GetFloat(), kDetScoreTol);
        matched[best] = true;
    }
}

void expect_keypoints(const std::string& name, const std_feature_point_output& points) {
    rapidjson::Document golden = load_golden_json(name);
    if (update_golden_mode()) {
        rapidjson::Document doc;
        doc.SetObject();
        rapidjson::Document::AllocatorType& a = doc.GetAllocator();
        rapidjson::Value arr(rapidjson::kArrayType);
        for (const auto& p : points) {
            rapidjson::Value pt(rapidjson::kArrayType);
            pt.PushBack(p.location.x, a);
            pt.PushBack(p.location.y, a);
            arr.PushBack(pt, a);
        }
        doc.AddMember("points", arr, a);
        write_golden_text(name, ".json", serialize_json(doc));
        GTEST_SKIP() << "golden updated: " << name;
    }
    ASSERT_TRUE(golden.IsObject() && golden.HasMember("points") && golden["points"].IsArray())
        << "golden missing, run with MORTRED_UPDATE_GOLDEN=1: " << name;

    const auto& golden_pts = golden["points"].GetArray();
    ASSERT_EQ(points.size(), golden_pts.Size()) << "keypoint count changed for " << name;
    int matched = 0;
    for (const auto& gp : golden_pts) {
        cv::Point2f target(gp[0].GetFloat(), gp[1].GetFloat());
        double min_dist = 1e9;
        for (const auto& p : points) {
            min_dist = std::min(min_dist, static_cast<double>(cv::norm(p.location - target)));
        }
        if (min_dist <= kKeypointMatchDist) ++matched;
    }
    EXPECT_GE(static_cast<double>(matched) / points.size(), 0.9)
        << "keypoint match ratio low for " << name;
}

void expect_embeddings(const std::string& name, const std::vector<float>& embeddings) {
    rapidjson::Document golden = load_golden_json(name);
    if (update_golden_mode()) {
        rapidjson::Document doc;
        doc.SetObject();
        rapidjson::Document::AllocatorType& a = doc.GetAllocator();
        rapidjson::Value arr(rapidjson::kArrayType);
        for (float e : embeddings) arr.PushBack(e, a);
        doc.AddMember("embeddings", arr, a);
        write_golden_text(name, ".json", serialize_json(doc));
        GTEST_SKIP() << "golden updated: " << name;
    }
    ASSERT_TRUE(golden.IsObject() && golden.HasMember("embeddings") && golden["embeddings"].IsArray())
        << "golden missing, run with MORTRED_UPDATE_GOLDEN=1: " << name;

    const auto& arr = golden["embeddings"].GetArray();
    ASSERT_EQ(embeddings.size(), arr.Size());
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    rapidjson::SizeType i = 0;
    for (const auto& e : arr) {
        dot += embeddings[i] * e.GetFloat();
        norm_a += embeddings[i] * embeddings[i];
        norm_b += e.GetFloat() * e.GetFloat();
        ++i;
    }
    double cosine = dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-9);
    EXPECT_GE(cosine, kEmbeddingCosThresh) << "embedding cosine similarity low for " << name;
}

bool weights_available(const std::string& conf_rel_path) {
    auto cfg = toml::parse(conf_rel_path);
    fix_toml_paths(cfg);
    std::vector<std::string> paths;
    std::function<void(const toml::value&)> collect = [&](const toml::value& v) {
        if (v.is_table()) {
            for (const auto& item : v.as_table()) {
                if (item.second.is_string() &&
                    (item.first == "model_file_path" || item.first == "vocab_file_path")) {
                    paths.push_back(item.second.as_string());
                }
                collect(item.second);
            }
        } else if (v.is_array()) {
            for (const auto& item : v.as_array()) collect(item);
        }
    };
    collect(cfg);
    for (const auto& p : paths) {
        if (!FilePathUtil::is_file_exist(p)) return false;
    }
    return true;
}

} // namespace

// ============ 分类（难度 1）============

TEST(model_golden, mobilenetv2_classification) {
    std::string conf = "conf/model/classification/mobilenetv2/mobilenetv2_config.ini";
    if (!weights_available(conf)) GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::classification::create_mobilenetv2_classifier<
        mat_input, std_classification_output>("mobilenetv2_golden");
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG");
    ASSERT_FALSE(image.empty());
    mat_input input{image};
    std_classification_output output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    expect_scores("mobilenetv2_classification", output);
}

TEST(model_golden, resnet50_classification) {
    std::string conf = "conf/model/classification/resnet/resnet50_config.ini";
    if (!weights_available(conf)) GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::classification::create_resnet_classifier<
        mat_input, std_classification_output>("resnet50_golden");
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG");
    ASSERT_FALSE(image.empty());
    mat_input input{image};
    std_classification_output output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    expect_scores("resnet50_classification", output);
}

TEST(model_golden, densenet121_classification) {
    std::string conf = "conf/model/classification/densenet/densenet121_config.ini";
    if (!weights_available(conf)) GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::classification::create_densenet_classifier<
        mat_input, std_classification_output>("densenet121_golden");
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG");
    ASSERT_FALSE(image.empty());
    mat_input input{image};
    std_classification_output output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    expect_scores("densenet121_classification", output);
}

TEST(model_golden, dinov2_classification) {
    std::string conf = "conf/model/classification/dinov2/dinov2_vitb14_config.ini";
    if (!weights_available(conf)) GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::classification::create_dinov2_classifier<
        mat_input, std_classification_output>("dinov2_golden");
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG");
    ASSERT_FALSE(image.empty());
    mat_input input{image};
    std_classification_output output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    expect_scores("dinov2_classification", output);
}

// ============ 检测（难度 2）============

TEST(model_golden, nanodet_detection) {
    std::string conf = "conf/model/object_detection/nano_det/nanodet_config.ini";
    if (!weights_available(conf)) GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::object_detection::create_nanodet_detector<
        mat_input, std_object_detection_output>("nanodet_golden");
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/object_detection/bus.jpg");
    ASSERT_FALSE(image.empty());
    mat_input input{image};
    std_object_detection_output output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    expect_boxes("nanodet_detection", output, false);
}

TEST(model_golden, yolov5_detection) {
    std::string conf = "conf/model/object_detection/yolov5/yolov5_config.ini";
    if (!weights_available(conf)) GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::object_detection::create_yolov5_detector<
        mat_input, std_object_detection_output>("yolov5_golden");
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/object_detection/bus.jpg");
    ASSERT_FALSE(image.empty());
    mat_input input{image};
    std_object_detection_output output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    expect_boxes("yolov5_detection", output, false);
}

TEST(model_golden, yolov6_detection) {
    std::string conf = "conf/model/object_detection/yolov6/yolov6_config.ini";
    if (!weights_available(conf)) GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::object_detection::create_yolov6_detector<
        mat_input, std_object_detection_output>("yolov6_golden");
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/object_detection/bus.jpg");
    ASSERT_FALSE(image.empty());
    mat_input input{image};
    std_object_detection_output output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    expect_boxes("yolov6_detection", output, false);
}

TEST(model_golden, yolov7_detection) {
    std::string conf = "conf/model/object_detection/yolov7/yolov7_config.ini";
    if (!weights_available(conf)) GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::object_detection::create_yolov7_detector<
        mat_input, std_object_detection_output>("yolov7_golden");
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/object_detection/bus.jpg");
    ASSERT_FALSE(image.empty());
    mat_input input{image};
    std_object_detection_output output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    expect_boxes("yolov7_detection", output, false);
}

TEST(model_golden, centerface_detection) {
    std::string conf = "conf/model/object_detection/centerface/centerface_config.ini";
    if (!weights_available(conf)) GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::object_detection::create_centerface_detector<
        mat_input, std_face_detection_output>("centerface_golden");
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/object_detection/face_w_mask.jpg");
    ASSERT_FALSE(image.empty());
    mat_input input{image};
    std_face_detection_output output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    expect_boxes("centerface_detection", output, true);
}

TEST(model_golden, libface_detection) {
    std::string conf = "conf/model/object_detection/libfacedetection/640x480_config.ini";
    if (!weights_available(conf)) GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::object_detection::create_libface_detector<
        mat_input, std_face_detection_output>("libface_golden");
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/object_detection/face_wo_mask.jpg");
    ASSERT_FALSE(image.empty());
    mat_input input{image};
    std_face_detection_output output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    expect_boxes("libface_detection", output, true);
}

// ============ 分割（难度 3）============

TEST(model_golden, bisenetv2_segmentation) {
    std::string conf = "conf/model/scene_segmentation/bisenetv2/bisenetv2_config.ini";
    if (!weights_available(conf)) GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::scene_segmentation::create_bisenetv2_segmentor<
        mat_input, std_scene_segmentation_output>("bisenetv2_golden");
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/scene_segmentation/cityscapes_test.png");
    ASSERT_FALSE(image.empty());
    mat_input input{image};
    std_scene_segmentation_output output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    expect_fingerprint("bisenetv2_segmentation", output.segmentation_result);
}

TEST(model_golden, pphuman_segmentation) {
    std::string conf = "conf/model/scene_segmentation/pphuman/pphuman_config.ini";
    if (!weights_available(conf)) GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::scene_segmentation::create_pphuman_segmentor<
        mat_input, std_scene_segmentation_output>("pphuman_golden");
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/scene_segmentation/human_image.jpg");
    ASSERT_FALSE(image.empty());
    mat_input input{image};
    std_scene_segmentation_output output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    expect_fingerprint("pphuman_segmentation", output.segmentation_result);
}

// ============ OCR（难度 3）============

TEST(model_golden, dbnet_text_detection) {
    std::string conf = "conf/model/ocr/db_text_detector/dbnet_config.ini";
    if (!weights_available(conf)) GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::ocr::create_dbtext_detector<
        mat_input, std_text_regions_output>("dbnet_golden");
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/ocr/railway_ticket.png");
    ASSERT_FALSE(image.empty());
    mat_input input{image};
    std_text_regions_output output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    expect_text_regions("dbnet_text_detection", output);
}

// ============ Matting（难度 4）============

TEST(model_golden, modnet_matting) {
    std::string conf = "conf/model/matting/modnet/modnet_config.ini";
    if (!weights_available(conf)) GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::matting::create_modnet_segmentor<
        mat_input, std_matting_output>("modnet_golden");
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/matting/matting_test.jpg");
    ASSERT_FALSE(image.empty());
    mat_input input{image};
    std_matting_output output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    expect_fingerprint("modnet_matting", output.matting_result);
}

TEST(model_golden, ppmatting_matting) {
    std::string conf = "conf/model/matting/ppmatting/ppmatting_config.ini";
    if (!weights_available(conf)) GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::matting::create_ppmatting_segmentor<
        mat_input, std_matting_output>("ppmatting_golden");
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/matting/matting_test.jpg");
    ASSERT_FALSE(image.empty());
    mat_input input{image};
    std_matting_output output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    expect_fingerprint("ppmatting_matting", output.matting_result);
}

// ============ 增强（难度 4）============

TEST(model_golden, enlightengan_enhancement) {
    std::string conf = "conf/model/enhancement/enlighten_gan/enlightengan.ini";
    if (!weights_available(conf)) GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::enhancement::create_enlightengan_enhancementor<
        mat_input, std_enhancement_output>("enlightengan_golden");
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/enhancement/low_light/lol_test_1.png");
    ASSERT_FALSE(image.empty());
    mat_input input{image};
    std_enhancement_output output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    expect_fingerprint("enlightengan_enhancement", output.enhancement_result);
}

TEST(model_golden, attentivegan_enhancement) {
    std::string conf = "conf/model/enhancement/attentive_gan_derain/attentive_gan_derain_config.ini";
    if (!weights_available(conf)) GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::enhancement::create_attentivegan_enhancementor<
        mat_input, std_enhancement_output>("attentivegan_golden");
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/enhancement/derain/test_1.png");
    ASSERT_FALSE(image.empty());
    mat_input input{image};
    std_enhancement_output output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    expect_fingerprint("attentivegan_enhancement", output.enhancement_result);
}

TEST(model_golden, realesrgan_enhancement) {
    std::string conf = "conf/model/enhancement/real_esrgan/real_esrgan.ini";
    if (!weights_available(conf)) GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::enhancement::create_realesrgan_enhancementor<
        mat_input, std_enhancement_output>("realesrgan_golden");
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/enhancement/real_esr/wolf_gray.jpg");
    ASSERT_FALSE(image.empty());
    mat_input input{image};
    std_enhancement_output output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    expect_fingerprint("realesrgan_enhancement", output.enhancement_result);
}

// ============ 特征点（难度 5）============

TEST(model_golden, superpoint_feature_point) {
    std::string conf = "conf/model/feature_point/superpoint/superpoint_config.ini";
    if (!weights_available(conf)) GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::feature_point::create_superpoint_extractor<
        mat_input, std_feature_point_output>("superpoint_golden");
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/feature_point/test.png");
    ASSERT_FALSE(image.empty());
    mat_input input{image};
    std_feature_point_output output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    expect_keypoints("superpoint_feature_point", output);
}

// ============ SAM（难度 6）============

TEST(model_golden, fastsam_segmentation) {
    std::string conf = "conf/model/segment_anything/fast_sam_s_config.ini";
    if (!weights_available(conf)) GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::segment_anything::create_fast_sam_segmentor<
        mat_input, cv::Mat>("fastsam_golden");
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/sam/truck.jpg");
    ASSERT_FALSE(image.empty());
    mat_input input{image};
    cv::Mat output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    expect_fingerprint("fastsam_segmentation", output);
}

// ============ CLIP（难度 7）============

TEST(model_golden, openai_clip_embedding) {
    std::string conf = "conf/model/openai_clip/vit_b_32_config.ini";
    if (!weights_available(conf)) GTEST_SKIP() << "weights not available";
    auto cfg = load_model_cfg(conf);
    auto model = jinq::factory::clip::create_openai_clip<
        jinq::models::io_define::clip::clip_input,
        jinq::models::io_define::clip::clip_output>("openai_clip_golden");
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->init(cfg), StatusCode::OK);
    cv::Mat image = read_input_image("demo_data/model_test_input/clip/fox.jpg");
    ASSERT_FALSE(image.empty());
    jinq::models::io_define::clip::clip_input input;
    input.task_type = jinq::models::io_define::clip::ClipTaskType::IMAGE_EMBEDDING;
    input.image = image;
    jinq::models::io_define::clip::clip_output output;
    ASSERT_EQ(model->run(input, output), StatusCode::OK);
    expect_embeddings("openai_clip_embedding", output.embeddings);
}
