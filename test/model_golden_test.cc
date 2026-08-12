/************************************************
 * Author: Codex
 * File: model_golden_test.cc
 * Date: 2026-08-12
 *
 * 模型推理黄金回归测试（L2）：
 * - 固定输入 + 固定配置运行真实模型，与 test/golden/ 中的基线输出比较，
 *   用于快速验证"修改模型推理函数后行为未改变"。
 * - 权重不在 git 仓库：权重缺失时 GTEST_SKIP（本地/GPU 机器才执行）。
 * - MORTRED_UPDATE_GOLDEN=1 时重新生成黄金文件（不比较）。
 * - MORTRED_MODEL_BACKEND 可指定后端（默认 cpu，保证可移植与确定性）。
 ************************************************/

#include <cstdlib>
#include <filesystem>
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
#include "models/model_io_define.h"

using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::models::io_define::classification::std_classification_output;
using jinq::models::io_define::common_io::mat_input;

namespace {

std::string get_weights_root() {
    const char* env = std::getenv("MORTRED_WEIGHTS_DIR");
    return env != nullptr && *env != '\0' ? std::string(env) : std::string("weights");
}

std::string get_golden_root() {
    return "test/golden";
}

bool update_golden_mode() {
    const char* env = std::getenv("MORTRED_UPDATE_GOLDEN");
    return env != nullptr && std::string(env) == "1";
}

toml::value build_mobilenetv2_cfg() {
    const char* backend_env = std::getenv("MORTRED_MODEL_BACKEND");
    std::string backend = backend_env != nullptr && *backend_env != '\0' ? backend_env : "cpu";

    toml::value cfg;
    cfg["MOBILENETV2"]["model_file_path"] =
        get_weights_root() + "/classification/mobilenetv2/mobilenetv2_ilsvrc2012.mnn";
    cfg["MOBILENETV2"]["model_threads_num"] = 1;
    cfg["MOBILENETV2"]["model_input_image_size"] = toml::array{224, 224};
    cfg["MOBILENETV2"]["compute_backend"] = backend;
    cfg["MOBILENETV2"]["backend_precision_mode"] = 0;
    cfg["MOBILENETV2"]["backend_power_mode"] = 0;
    cfg["MOBILENETV2"]["class_name_file"] = "conf/model/classification/imagenet_classes.txt";
    return cfg;
}

std::string serialize_golden(const std_classification_output& output) {
    rapidjson::Document doc;
    doc.SetObject();
    rapidjson::Document::AllocatorType& allocator = doc.GetAllocator();
    doc.AddMember("class_id", output.class_id, allocator);
    rapidjson::Value scores(rapidjson::kArrayType);
    for (float score : output.scores) {
        scores.PushBack(score, allocator);
    }
    doc.AddMember("scores", scores, allocator);
    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    return buffer.GetString();
}

bool load_golden(const std::string& path, int* class_id, std::vector<float>* scores) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return false;
    }
    std::stringstream ss;
    ss << in.rdbuf();

    rapidjson::Document doc;
    doc.Parse(ss.str().c_str());
    if (doc.HasParseError() || !doc.IsObject() || !doc.HasMember("class_id") ||
        !doc.HasMember("scores") || !doc["scores"].IsArray()) {
        return false;
    }
    *class_id = doc["class_id"].GetInt();
    scores->clear();
    for (const auto& value : doc["scores"].GetArray()) {
        scores->push_back(value.GetFloat());
    }
    return true;
}

void write_golden(const std::string& path, const std::string& content) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path);
    out << content;
}

} // namespace

TEST(model_golden, mobilenetv2_classification) {
    auto cfg = build_mobilenetv2_cfg();
    std::string model_path = cfg["MOBILENETV2"]["model_file_path"].as_string();
    if (!FilePathUtil::is_file_exist(model_path)) {
        GTEST_SKIP() << "weights not available: " << model_path;
    }

    auto classifier = jinq::factory::classification::create_mobilenetv2_classifier<
        mat_input, std_classification_output>("mobilenetv2_golden");
    ASSERT_NE(classifier, nullptr);
    ASSERT_EQ(classifier->init(cfg), StatusCode::OK);
    ASSERT_TRUE(classifier->is_successfully_initialized());

    cv::Mat image = cv::imread(
        "demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG",
        cv::IMREAD_COLOR);
    ASSERT_FALSE(image.empty());

    mat_input model_input{image};
    std_classification_output output;
    ASSERT_EQ(classifier->run(model_input, output), StatusCode::OK);

    std::string golden_path = get_golden_root() + "/mobilenetv2_classification.json";
    if (update_golden_mode()) {
        write_golden(golden_path, serialize_golden(output));
        GTEST_SKIP() << "golden updated: " << golden_path;
    }

    int golden_class_id = -1;
    std::vector<float> golden_scores;
    ASSERT_TRUE(load_golden(golden_path, &golden_class_id, &golden_scores))
        << "golden file missing, run with MORTRED_UPDATE_GOLDEN=1 first: " << golden_path;

    EXPECT_EQ(output.class_id, golden_class_id);
    ASSERT_EQ(output.scores.size(), golden_scores.size());
    for (size_t i = 0; i < output.scores.size(); ++i) {
        EXPECT_NEAR(output.scores[i], golden_scores[i], 1e-3)
            << "score mismatch at index " << i;
    }
}
