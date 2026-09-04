/************************************************
 * Author: Codex
 * File: model_golden_registry.h
 * Date: 26-8-30
 *
 * Golden case registry: one macro per case, no per-case plumbing.
 *
 * A standard golden case always does the same seven steps - check weights,
 * load + normalise the config, create the model, init, read the input image,
 * run, compare against test/golden/. The registry owns those steps; a case now
 * only declares its identity (name, config, image, creator) and how its output
 * is compared.
 *
 * The macros expand to plain TEST(model_golden, name), so case names,
 * --gtest_filter behaviour and MORTRED_UPDATE_GOLDEN=1 are unchanged.
 * Cases that are genuinely different (batch equivalence, SAM prompt input,
 * CLIP text+image towers) stay hand-written in model_golden_test.cc instead of
 * being forced through a type-erased hole.
 ************************************************/

#ifndef MORTRED_TEST_MODEL_GOLDEN_REGISTRY_H
#define MORTRED_TEST_MODEL_GOLDEN_REGISTRY_H

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "ci_require_weights.h"
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <toml/toml.hpp>

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "common/status_code.h"
#include "models/base_model.h"
#include "models/io/classification.h"
#include "models/io/common_input.h"
#include "models/io/enhancement.h"
#include "models/io/feature_embedding.h"
#include "models/io/feature_point.h"
#include "models/io/matting.h"
#include "models/io/object_detection.h"
#include "models/io/ocr.h"
#include "models/io/scene_segmentation.h"

namespace jinq {
namespace test {
namespace golden {

using jinq::common::CvUtils;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::models::BaseAiModel;
using jinq::models::io_define::classification::std_classification_output;
using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::enhancement::std_enhancement_output;
using jinq::models::io_define::feature_embedding::std_feature_embedding_output;
using jinq::models::io_define::feature_point::std_feature_point_output;
using jinq::models::io_define::matting::std_matting_output;
using jinq::models::io_define::object_detection::bbox;
using jinq::models::io_define::object_detection::face_bbox;
using jinq::models::io_define::object_detection::std_face_detection_output;
using jinq::models::io_define::object_detection::std_object_detection_output;
using jinq::models::io_define::ocr::std_text_regions_output;
using jinq::models::io_define::scene_segmentation::std_scene_segmentation_output;

constexpr double k_score_tol = 1e-3;
constexpr double k_det_score_tol = 1e-2;
constexpr float k_box_iou_thresh = 0.5f;
constexpr double k_fingerprint_diff = 1.0;
constexpr double k_keypoint_match_dist = 3.0;
constexpr double k_embedding_cos_thresh = 0.999;

bool update_golden_mode() {
    const char *env = std::getenv("MORTRED_UPDATE_GOLDEN");
    return env != nullptr && std::string(env) == "1";
}

std::string golden_path(const std::string &name, const std::string &ext) { return "test/golden/" + name + ext; }

/*** ????? conf ???????????? cpu ?? */
void fix_toml_paths(toml::node &value) {
    if (auto *tbl = value.as_table()) {
        for (auto &item : *tbl) {
            if (item.second.is_string()) {
                std::string s = item.second.value_or<std::string>("");
                if (s.rfind("../", 0) == 0) {
                    item.second.ref<std::string>() = s.substr(3);
                }
            } else {
                fix_toml_paths(item.second);
            }
        }
    } else if (auto *arr = value.as_array()) {
        for (auto &item : *arr) {
            fix_toml_paths(item);
        }
    }
}

void force_cpu_backend(toml::node &value) {
    if (auto *tbl = value.as_table()) {
        for (auto &item : *tbl) {
            // new schema: device inside [SECTION.backend]; old schema: compute_backend
            if (item.first == "backend" && item.second.is_table()) {
                auto &backend_table = item.second.ref<toml::table>();
                if (backend_table.contains("device")) {
                    backend_table["device"].ref<std::string>() = std::string("cpu");
                }
            } else if (item.first == "compute_backend") {
                item.second.ref<std::string>() = std::string("cpu");
            } else {
                force_cpu_backend(item.second);
            }
        }
    } else if (auto *arr = value.as_array()) {
        for (auto &item : *arr) {
            force_cpu_backend(item);
        }
    }
}

toml::table load_model_cfg(const std::string &conf_rel_path) {
    auto cfg_parsed = toml::parse_file(conf_rel_path);
    if (!cfg_parsed) {
        LOG(ERROR) << "parse model config file failed: " << conf_rel_path << ", error: " << std::string(cfg_parsed.error().description());
        return toml::table{};
    }
    auto cfg = std::move(cfg_parsed).table();
    fix_toml_paths(cfg);
    force_cpu_backend(cfg);
    return cfg;
}

cv::Mat read_input_image(const std::string &path) { return cv::imread(path, cv::IMREAD_COLOR); }

/*** ?? JSON ?? */
std::string serialize_json(const rapidjson::Document &doc) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    return buffer.GetString();
}

rapidjson::Document load_golden_json(const std::string &name) {
    rapidjson::Document doc;
    std::ifstream in(golden_path(name, ".json"));
    if (in.is_open()) {
        std::stringstream ss;
        ss << in.rdbuf();
        doc.Parse(ss.str().c_str());
    }
    return doc;
}

void write_golden_text(const std::string &name, const std::string &ext, const std::string &content) {
    std::string path = golden_path(name, ext);
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path);
    out << content;
}

/*** ????????? Mat -> 64x64 CV_8UC3???????? */
cv::Mat make_fingerprint(const cv::Mat &src) {
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

void save_golden_fingerprint(const std::string &name, const cv::Mat &mat) {
    std::string path = golden_path(name, ".png");
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    cv::imwrite(path, make_fingerprint(mat));
}

void expect_fingerprint(const std::string &name, const cv::Mat &mat) {
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
    EXPECT_LE(mean, k_fingerprint_diff) << "fingerprint drift for " << name << ", mean abs diff = " << mean;
}

void expect_scores(const std::string &name, const std_classification_output &output) {
    rapidjson::Document golden = load_golden_json(name);
    if (update_golden_mode()) {
        rapidjson::Document doc;
        doc.SetObject();
        rapidjson::Document::AllocatorType &a = doc.GetAllocator();
        doc.AddMember("class_id", output.class_id, a);
        rapidjson::Value scores(rapidjson::kArrayType);
        for (float s : output.scores)
            scores.PushBack(s, a);
        doc.AddMember("scores", scores, a);
        write_golden_text(name, ".json", serialize_json(doc));
        GTEST_SKIP() << "golden updated: " << name;
    }
    ASSERT_TRUE(golden.IsObject() && golden.HasMember("class_id") && golden.HasMember("scores"))
        << "golden missing, run with MORTRED_UPDATE_GOLDEN=1: " << name;
    EXPECT_EQ(output.class_id, golden["class_id"].GetInt());
    ASSERT_EQ(output.scores.size(), golden["scores"].GetArray().Size());
    size_t idx = 0;
    for (const auto &s : golden["scores"].GetArray()) {
        EXPECT_NEAR(output.scores[idx], s.GetFloat(), k_score_tol) << "score mismatch at " << idx;
        ++idx;
    }
}

const std::vector<cv::Point2f> &get_landmarks(const face_bbox &box) { return box.landmarks; }

const std::vector<cv::Point2f> &get_landmarks(const bbox &) {
    static const std::vector<cv::Point2f> k_empty;
    return k_empty;
}

template <typename BoxT> rapidjson::Value serialize_box(const BoxT &box, rapidjson::Document::AllocatorType &a) {
    rapidjson::Value obj(rapidjson::kObjectType);
    obj.AddMember("x", box.bbox.x, a);
    obj.AddMember("y", box.bbox.y, a);
    obj.AddMember("w", box.bbox.width, a);
    obj.AddMember("h", box.bbox.height, a);
    obj.AddMember("score", box.score, a);
    obj.AddMember("class_id", box.class_id, a);
    return obj;
}

template <typename BoxT> void expect_boxes(const std::string &name, const std::vector<BoxT> &boxes, bool has_landmarks) {
    rapidjson::Document golden = load_golden_json(name);
    if (update_golden_mode()) {
        rapidjson::Document doc;
        doc.SetObject();
        rapidjson::Document::AllocatorType &a = doc.GetAllocator();
        rapidjson::Value arr(rapidjson::kArrayType);
        for (const auto &box : boxes) {
            rapidjson::Value obj = serialize_box(box, a);
            if (has_landmarks) {
                rapidjson::Value pts(rapidjson::kArrayType);
                for (const auto &p : get_landmarks(box)) {
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

    const auto &golden_boxes = golden["boxes"].GetArray();
    ASSERT_EQ(boxes.size(), golden_boxes.Size()) << "detection count changed for " << name;

    std::vector<bool> matched(golden_boxes.Size(), false);
    for (const auto &box : boxes) {
        int best = -1;
        float best_iou = 0.0f;
        for (rapidjson::SizeType i = 0; i < golden_boxes.Size(); ++i) {
            if (matched[i])
                continue;
            const auto &g = golden_boxes[i];
            cv::Rect2f gbox(g["x"].GetFloat(), g["y"].GetFloat(), g["w"].GetFloat(), g["h"].GetFloat());
            float iou = CvUtils::calc_iou(box.bbox, gbox);
            if (iou > best_iou) {
                best_iou = iou;
                best = static_cast<int>(i);
            }
        }
        ASSERT_GE(best, 0) << "unmatched detection for " << name;
        EXPECT_GE(best_iou, k_box_iou_thresh) << "low IoU for " << name;
        const auto &g = golden_boxes[best];
        EXPECT_NEAR(box.score, g["score"].GetFloat(), k_det_score_tol);
        EXPECT_EQ(box.class_id, g["class_id"].GetInt());
        if (has_landmarks && g.HasMember("landmarks")) {
            ASSERT_EQ(get_landmarks(box).size(), g["landmarks"].GetArray().Size());
            rapidjson::SizeType li = 0;
            for (const auto &lp : g["landmarks"].GetArray()) {
                cv::Point2f gp(lp[0].GetFloat(), lp[1].GetFloat());
                EXPECT_LE(cv::norm(get_landmarks(box)[li] - gp), k_keypoint_match_dist);
                ++li;
            }
        }
        matched[best] = true;
    }
}

template <typename BoxT>
void expect_equivalent_detections(const std::string &name, const std::vector<BoxT> &expected, const std::vector<BoxT> &actual) {
    ASSERT_EQ(expected.size(), actual.size()) << name;
    for (size_t idx = 0; idx < expected.size(); ++idx) {
        EXPECT_EQ(expected[idx].class_id, actual[idx].class_id) << name << " item " << idx;
        EXPECT_NEAR(expected[idx].score, actual[idx].score, 1e-3) << name << " item " << idx;
        EXPECT_GE(CvUtils::calc_iou(expected[idx].bbox, actual[idx].bbox), 0.995f) << name << " item " << idx;
    }
}

void expect_text_regions(const std::string &name, const std_text_regions_output &regions) {
    rapidjson::Document golden = load_golden_json(name);
    if (update_golden_mode()) {
        rapidjson::Document doc;
        doc.SetObject();
        rapidjson::Document::AllocatorType &a = doc.GetAllocator();
        rapidjson::Value arr(rapidjson::kArrayType);
        for (const auto &region : regions) {
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

    const auto &golden_regions = golden["regions"].GetArray();
    ASSERT_EQ(regions.size(), golden_regions.Size()) << "text region count changed for " << name;
    std::vector<bool> matched(golden_regions.Size(), false);
    for (const auto &region : regions) {
        int best = -1;
        float best_iou = 0.0f;
        for (rapidjson::SizeType i = 0; i < golden_regions.Size(); ++i) {
            if (matched[i])
                continue;
            const auto &g = golden_regions[i];
            cv::Rect2f gbox(g["x"].GetFloat(), g["y"].GetFloat(), g["w"].GetFloat(), g["h"].GetFloat());
            float iou = CvUtils::calc_iou(region.bbox, gbox);
            if (iou > best_iou) {
                best_iou = iou;
                best = static_cast<int>(i);
            }
        }
        ASSERT_GE(best, 0) << "unmatched text region for " << name;
        EXPECT_GE(best_iou, k_box_iou_thresh) << "low IoU for " << name;
        EXPECT_NEAR(region.score, golden_regions[best]["score"].GetFloat(), k_det_score_tol);
        matched[best] = true;
    }
}

void expect_keypoints(const std::string &name, const std_feature_point_output &points) {
    rapidjson::Document golden = load_golden_json(name);
    if (update_golden_mode()) {
        rapidjson::Document doc;
        doc.SetObject();
        rapidjson::Document::AllocatorType &a = doc.GetAllocator();
        rapidjson::Value arr(rapidjson::kArrayType);
        for (const auto &p : points) {
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

    const auto &golden_pts = golden["points"].GetArray();
    ASSERT_EQ(points.size(), golden_pts.Size()) << "keypoint count changed for " << name;
    int matched = 0;
    for (const auto &gp : golden_pts) {
        cv::Point2f target(gp[0].GetFloat(), gp[1].GetFloat());
        double min_dist = 1e9;
        for (const auto &p : points) {
            min_dist = std::min(min_dist, static_cast<double>(cv::norm(p.location - target)));
        }
        if (min_dist <= k_keypoint_match_dist)
            ++matched;
    }
    EXPECT_GE(static_cast<double>(matched) / points.size(), 0.9) << "keypoint match ratio low for " << name;
}

void expect_embeddings(const std::string &name, const std::vector<float> &embeddings) {
    rapidjson::Document golden = load_golden_json(name);
    if (update_golden_mode()) {
        rapidjson::Document doc;
        doc.SetObject();
        rapidjson::Document::AllocatorType &a = doc.GetAllocator();
        rapidjson::Value arr(rapidjson::kArrayType);
        for (float e : embeddings)
            arr.PushBack(e, a);
        doc.AddMember("embeddings", arr, a);
        write_golden_text(name, ".json", serialize_json(doc));
        GTEST_SKIP() << "golden updated: " << name;
    }
    ASSERT_TRUE(golden.IsObject() && golden.HasMember("embeddings") && golden["embeddings"].IsArray())
        << "golden missing, run with MORTRED_UPDATE_GOLDEN=1: " << name;

    const auto &arr = golden["embeddings"].GetArray();
    ASSERT_EQ(embeddings.size(), arr.Size());
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    rapidjson::SizeType i = 0;
    for (const auto &e : arr) {
        dot += embeddings[i] * e.GetFloat();
        norm_a += embeddings[i] * embeddings[i];
        norm_b += e.GetFloat() * e.GetFloat();
        ++i;
    }
    double cosine = dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-9);
    EXPECT_GE(cosine, k_embedding_cos_thresh) << "embedding cosine similarity low for " << name;
}

bool weights_available(const std::string &conf_rel_path) {
    auto cfg_parsed = toml::parse_file(conf_rel_path);
    if (!cfg_parsed) {
        LOG(ERROR) << "parse model config file failed: " << conf_rel_path << ", error: " << std::string(cfg_parsed.error().description());
        return false;
    }
    auto cfg = std::move(cfg_parsed).table();
    fix_toml_paths(cfg);
    std::vector<std::string> paths;
    std::function<void(const toml::node &)> collect = [&](const toml::node &v) {
        if (const auto *tbl = v.as_table()) {
            for (const auto &item : *tbl) {
                if (item.second.is_string() && (item.first == "model_file_path" || item.first == "vocab_file_path")) {
                    paths.push_back(item.second.value_or<std::string>(""));
                }
                collect(item.second);
            }
        } else if (const auto *arr = v.as_array()) {
            for (const auto &item : *arr)
                collect(item);
        }
    };
    collect(cfg);
    for (const auto &p : paths) {
        if (!FilePathUtil::is_file_exist(p))
            return false;
    }
    return true;
}

/*** typed comparators: one line per task output contract ***/

inline void compare_classification(const std::string &name, const std_classification_output &out) { expect_scores(name, out); }

inline void compare_feature_embedding(const std::string &name, const std_feature_embedding_output &out) {
    expect_embeddings(name, out.embedding);
}

inline void compare_object_detection(const std::string &name, const std_object_detection_output &out) { expect_boxes(name, out, false); }

inline void compare_face_detection(const std::string &name, const std_face_detection_output &out) { expect_boxes(name, out, true); }

inline void compare_scene_segmentation(const std::string &name, const std_scene_segmentation_output &out) {
    expect_fingerprint(name, out.segmentation_result);
}

inline void compare_matting(const std::string &name, const std_matting_output &out) { expect_fingerprint(name, out.matting_result); }

inline void compare_enhancement(const std::string &name, const std_enhancement_output &out) {
    expect_fingerprint(name, out.enhancement_result);
}

inline void compare_text_regions(const std::string &name, const std_text_regions_output &out) { expect_text_regions(name, out); }

inline void compare_keypoints(const std::string &name, const std_feature_point_output &out) { expect_keypoints(name, out); }

inline void compare_raw_mat(const std::string &name, const cv::Mat &out) { expect_fingerprint(name, out); }

/*** the seven shared steps every standard golden case performs ***/

template <typename OUTPUT> using GoldenCreator = std::unique_ptr<BaseAiModel<mat_input, OUTPUT>> (*)();

template <typename OUTPUT> using GoldenCompare = void (*)(const std::string &, const OUTPUT &);

template <typename OUTPUT>
void run_case(const char *name, const char *config, const char *image, GoldenCreator<OUTPUT> creator, GoldenCompare<OUTPUT> compare) {
    const std::string conf(config);
    if (!weights_available(conf)) {
        MORTRED_SKIP_OR_FAIL_WEIGHTS("weights not available");
    }
    auto cfg = load_model_cfg(conf);
    auto model = creator();
    ASSERT_NE(model, nullptr) << name;
    ASSERT_EQ(model->init(cfg), StatusCode::OK) << name;
    const cv::Mat input_image = read_input_image(image);
    ASSERT_FALSE(input_image.empty()) << name << ": " << image;
    OUTPUT output;
    ASSERT_EQ(model->run(mat_input{input_image}, output), StatusCode::OK) << name;
    SCOPED_TRACE(name);
    compare(name, output);
}

} // namespace golden
} // namespace test
} // namespace jinq

/*** one macro per task output contract - no type erasure.
 *
 * The creator is passed unspecialised together with its OUTPUT type: a single
 * macro argument must not contain a comma, and `creator<mat_input, OUTPUT>`
 * would otherwise be split into two arguments by the preprocessor. ***/

#define MORTRED_GOLDEN_CASE(name, config, image, creator, output_type, compare_fn)                                                         \
    TEST(model_golden, name) {                                                                                                             \
        /* the case name is an identifier for TEST() but a string for the  */                                                              \
        /* golden stem, the log tag and every failure message              */                                                              \
        jinq::test::golden::run_case(#name, config, image, &creator<jinq::models::io_define::common_io::mat_input, output_type>,           \
                                     jinq::test::golden::compare_fn);                                                                      \
    }

#define GOLDEN_CLASSIFICATION_CASE(name, config, image, creator, output_type)                                                              \
    MORTRED_GOLDEN_CASE(name, config, image, creator, output_type, compare_classification)

#define GOLDEN_FEATURE_EMBEDDING_CASE(name, config, image, creator, output_type)                                                           \
    MORTRED_GOLDEN_CASE(name, config, image, creator, output_type, compare_feature_embedding)

#define GOLDEN_OBJECT_DETECTION_CASE(name, config, image, creator, output_type)                                                            \
    MORTRED_GOLDEN_CASE(name, config, image, creator, output_type, compare_object_detection)

#define GOLDEN_FACE_DETECTION_CASE(name, config, image, creator, output_type)                                                              \
    MORTRED_GOLDEN_CASE(name, config, image, creator, output_type, compare_face_detection)

#define GOLDEN_SCENE_SEGMENTATION_CASE(name, config, image, creator, output_type)                                                          \
    MORTRED_GOLDEN_CASE(name, config, image, creator, output_type, compare_scene_segmentation)

#define GOLDEN_MATTING_CASE(name, config, image, creator, output_type)                                                                     \
    MORTRED_GOLDEN_CASE(name, config, image, creator, output_type, compare_matting)

#define GOLDEN_ENHANCEMENT_CASE(name, config, image, creator, output_type)                                                                 \
    MORTRED_GOLDEN_CASE(name, config, image, creator, output_type, compare_enhancement)

#define GOLDEN_TEXT_REGION_CASE(name, config, image, creator, output_type)                                                                 \
    MORTRED_GOLDEN_CASE(name, config, image, creator, output_type, compare_text_regions)

#define GOLDEN_KEYPOINT_CASE(name, config, image, creator, output_type)                                                                    \
    MORTRED_GOLDEN_CASE(name, config, image, creator, output_type, compare_keypoints)

#define GOLDEN_RAW_MAT_CASE(name, config, image, creator, output_type)                                                                     \
    MORTRED_GOLDEN_CASE(name, config, image, creator, output_type, compare_raw_mat)

#endif // MORTRED_TEST_MODEL_GOLDEN_REGISTRY_H
