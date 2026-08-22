/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: trt_engine_inspect.cc
* Date: 26-8-22
************************************************/

// Engine diagnostic: prints every IO tensor with the shape TensorRT derived
// at build time. A batch-profile engine prints [-1,C,H,W]; a static engine
// prints [1,C,H,W] - one glance tells whether a rebuild actually took.

#include <NvInfer.h>

#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
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
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: trt_engine_inspect engine" << std::endl;
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    std::vector<char> buf((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
    std::unique_ptr<IRuntime> runtime(createInferRuntime(g_logger));
    std::unique_ptr<ICudaEngine> engine(
        runtime->deserializeCudaEngine(buf.data(), buf.size()));
    if (engine == nullptr) {
        std::cerr << "deserialize failed: " << argv[1] << std::endl;
        return 3;
    }
    const int32_t nb_io = engine->getNbIOTensors();
    std::cout << "io tensors (" << nb_io << "):" << std::endl;
    for (int32_t i = 0; i < nb_io; ++i) {
        const char* name = engine->getIOTensorName(i);
        const auto shape = engine->getTensorShape(name);
        std::cout << "  " << name << " [";
        for (int32_t d = 0; d < shape.nbDims; ++d) {
            std::cout << (d != 0 ? "," : "") << shape.d[d];
        }
        std::cout << "]" << std::endl;
        if (engine->getTensorShape(name).nbDims > 0) {
            const bool is_input = engine->getTensorIOMode(name) == TensorIOMode::kINPUT;
            if (is_input) {
                for (const auto sel : {OptProfileSelector::kMIN, OptProfileSelector::kOPT,
                                       OptProfileSelector::kMAX}) {
                    const auto prof = engine->getProfileShape(name, 0, sel);
                    if (prof.nbDims <= 0) {
                        continue;
                    }
                    const char* tag = sel == OptProfileSelector::kMIN
                                          ? "min"
                                          : sel == OptProfileSelector::kOPT ? "opt" : "max";
                    std::cout << "    profile " << tag << " [";
                    for (int32_t d = 0; d < prof.nbDims; ++d) {
                        std::cout << (d != 0 ? "," : "") << prof.d[d];
                    }
                    std::cout << "]" << std::endl;
                }
            }
        }
    }
    return 0;
}
