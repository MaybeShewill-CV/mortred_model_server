/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: trt_engine_builder.cc
* Date: 26-8-22
************************************************/

// Minimal trtexec replacement for boxes without the TensorRT CLI: parses an
// ONNX model and builds an engine with explicit-batch optimization profiles.
// Semantics intentionally mirror the flags consumed by
// scripts/convert_trt_engines.sh:
//
//   trt_engine_builder --onnx model.onnx --save out.engine [--fp16] \
//       --min images:1x3x640x640 --opt images:8x3x640x640 --max images:16x3x640x640

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace nvinfer1;

namespace {

struct Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[trt] " << msg << std::endl;
        }
    }
};

Logger g_logger;

bool parse_named_dims(const std::string& arg, std::string* name, std::vector<int32_t>* dims) {
    const auto colon = arg.find(':');
    if (colon == std::string::npos) {
        return false;
    }
    *name = arg.substr(0, colon);
    const std::string rest = arg.substr(colon + 1);
    size_t pos = 0;
    while (pos < rest.size()) {
        const auto x = rest.find('x', pos);
        const auto piece = rest.substr(pos, x == std::string::npos ? std::string::npos : x - pos);
        dims->push_back(static_cast<int32_t>(std::atoi(piece.c_str())));
        if (x == std::string::npos) {
            break;
        }
        pos = x + 1;
    }
    return !name->empty() && !dims->empty();
}

Dims to_dims(const std::vector<int32_t>& v) {
    Dims d{};
    d.nbDims = static_cast<int32_t>(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        d.d[i] = v[i];
    }
    return d;
}

}  // namespace

int main(int argc, char** argv) {
    std::string onnx_path;
    std::string save_path;
    bool fp16 = false;
    std::unordered_map<std::string, Dims> shape_min;
    std::unordered_map<std::string, Dims> shape_opt;
    std::unordered_map<std::string, Dims> shape_max;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool has_value = i + 1 < argc;
        if (arg == "--onnx" && has_value) {
            onnx_path = argv[++i];
        } else if (arg == "--save" && has_value) {
            save_path = argv[++i];
        } else if (arg == "--fp16") {
            fp16 = true;
        } else if ((arg == "--min" || arg == "--opt" || arg == "--max") && has_value) {
            std::string name;
            std::vector<int32_t> dims;
            if (!parse_named_dims(argv[++i], &name, &dims)) {
                std::cerr << "bad shape spec (expected name:d1xd2x...): " << argv[i] << std::endl;
                return 2;
            }
            (arg == "--min" ? shape_min : arg == "--opt" ? shape_opt : shape_max)[name] =
                to_dims(dims);
        } else {
            std::cerr << "unknown or incomplete argument: " << arg << std::endl;
            return 2;
        }
    }
    if (onnx_path.empty() || save_path.empty()) {
        std::cerr << "usage: trt_engine_builder --onnx X --save Y [--fp16] "
                     "--min/--opt/--max name:d1xd2x..."
                  << std::endl;
        return 2;
    }

    std::unique_ptr<IBuilder> builder(createInferBuilder(g_logger));
    if (builder == nullptr) {
        std::cerr << "createInferBuilder failed (CUDA/GPU available?)" << std::endl;
        return 3;
    }
    std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(
        1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    std::unique_ptr<nvonnxparser::IParser> parser(
        nvonnxparser::createParser(*network, g_logger));
    if (network == nullptr || parser == nullptr) {
        std::cerr << "create network/parser failed" << std::endl;
        return 3;
    }
    if (!parser->parseFromFile(onnx_path.c_str(),
                               static_cast<int>(ILogger::Severity::kWARNING))) {
        std::cerr << "parse onnx failed: " << onnx_path << std::endl;
        return 3;
    }
    for (int32_t i = 0; i < network->getNbInputs(); ++i) {
        const auto* input = network->getInput(i);
        const auto dims = input->getDimensions();
        std::cout << "network input '" << input->getName() << "' [";
        for (int32_t d = 0; d < dims.nbDims; ++d) {
            std::cout << (d != 0 ? "," : "") << dims.d[d];
        }
        std::cout << "]" << std::endl;
    }

    std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
    if (config == nullptr) {
        std::cerr << "createBuilderConfig failed" << std::endl;
        return 3;
    }
    if (fp16) {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (!shape_min.empty()) {
        IOptimizationProfile* profile = builder->createOptimizationProfile();
        for (const auto& kv : shape_min) {
            const auto opt = shape_opt.find(kv.first);
            const auto max = shape_max.find(kv.first);
            if (opt == shape_opt.end() || max == shape_max.end()) {
                std::cerr << "input '" << kv.first << "' needs --min, --opt and --max" << std::endl;
                return 2;
            }
            if (!profile->setDimensions(kv.first.c_str(), OptProfileSelector::kMIN, kv.second) ||
                !profile->setDimensions(kv.first.c_str(), OptProfileSelector::kOPT, opt->second) ||
                !profile->setDimensions(kv.first.c_str(), OptProfileSelector::kMAX, max->second)) {
                std::cerr << "setDimensions failed for input '" << kv.first << "'" << std::endl;
                return 3;
            }
        }
        if (config->addOptimizationProfile(profile) == -1) {
            std::cerr << "addOptimizationProfile failed" << std::endl;
            return 3;
        }
    }

    std::cout << "building engine (fp16=" << fp16 << ", profiled inputs=" << shape_min.size()
              << ") ..." << std::endl;
    std::unique_ptr<IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    if (serialized == nullptr) {
        std::cerr << "buildSerializedNetwork failed" << std::endl;
        return 3;
    }
    std::ofstream out(save_path, std::ios::binary);
    out.write(static_cast<const char*>(serialized->data()),
              static_cast<std::streamsize>(serialized->size()));
    if (!out.good()) {
        std::cerr << "write engine failed: " << save_path << std::endl;
        return 3;
    }
    std::cout << "engine saved: " << save_path << " (" << serialized->size() << " bytes)"
              << std::endl;
    return 0;
}
