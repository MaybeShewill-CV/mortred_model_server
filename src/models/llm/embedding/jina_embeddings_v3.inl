/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: jina_embeddings_v3.inl
 * Date: 24-12-17
 ************************************************/

#include "jina_embeddings_v3.h"

#include <glog/logging.h>
#include "fmt/format.h"
#include "onnxruntime/onnxruntime_cxx_api.h"
#include "TensorRT-8.6.1.6/NvInferRuntime.h"

#include "common/cv_utils.h"
#include "common/time_stamp.h"
#include "common/file_path_util.h"
#include "models/llm/llama/llama3.h"
#include "models/trt_helper/trt_helper.h"

namespace jinq {
namespace models {
namespace llm {

using jinq::common::CvUtils;
using jinq::common::Timestamp;
using jinq::common::StatusCode;
using jinq::common::FilePathUtil;
using jinq::models::io_define::llm::embedding::pool_type;
using jinq::models::io_define::llm::embedding::std_embedding_input;
using jinq::models::io_define::llm::embedding::std_embedding_output;
using TokenizerPtr = jinq::models::llm::llama::Llama3<std::string, std::string>;

namespace embedding {

using trt_helper::EngineBinding;
using trt_helper::DeviceMemory;
using trt_helper::TrtHelper;
using trt_helper::TrtLogger;

namespace jina_embedding_impl {

using internal_input = std_embedding_input;
using internal_output = std_embedding_output;

/***
*
* @tparam INPUT
* @param in
* @return
 */
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, char*>::value, internal_input>::type transform_input(const INPUT& in) {
    std::string text = std::string(in);
    internal_input out {text, pool_type::EMBEDDING_MEAN_POOLING};
    return out;
}

/***
*
* @tparam INPUT
* @param in
* @return
 */
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, std::string>::value, internal_input>::type transform_input(const INPUT& in) {
    internal_input out {in, pool_type::EMBEDDING_MEAN_POOLING};
    return out;
}

/***
*
* @tparam INPUT
* @param in
* @return
 */
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, std_embedding_input >::value, internal_input>::type transform_input(const INPUT& in) {
    return in;
}

/***
* transform different type of internal output into external output
* @tparam EXTERNAL_OUTPUT
* @tparam dummy
* @param in
* @return
 */
template<typename OUTPUT>
typename std::enable_if<std::is_same<OUTPUT, std::decay<std_embedding_output >::type>::value, std_embedding_output>::type
transform_output(const jina_embedding_impl::internal_output& internal_out) {
    return internal_out;
}

}

/***************** Impl Function Sets ******************/

template <typename INPUT, typename OUTPUT>
class JinaEmbeddingsV3<INPUT, OUTPUT>::Impl {
  public:
    /***
     *
     */
    Impl() = default;

    /***
     *
     */
    ~Impl() {
        if (_m_backend_type == TRT) {
            LOG(ERROR) << "trt backend has not been implemented";
        } else {
            if (nullptr != _m_onnx_params.session) {
                _m_onnx_params.session->release();
            }
        }
    }

    /***
    *
    * @param transformer
     */
    Impl(const Impl& transformer) = delete;

    /***
     *
     * @param transformer
     * @return
     */
    Impl& operator=(const Impl& transformer) = delete;

    /***
     *
     * @param cfg_file_path
     * @return
     */
    StatusCode init(const decltype(toml::parse("")) &config);

    /***
     *
     * @param in
     * @param out
     * @return
     */
    StatusCode run(const INPUT& in, OUTPUT& out);

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_initialized;
    };

  private:
    struct ONNXParams {
        std::string model_file_path;
        // ort env params
        int thread_nums = 1;
        std::string device = "cuda";
        int device_id = 0;
        Ort::Env env;
        Ort::SessionOptions session_options;
        Ort::Session* session = nullptr;
        Ort::AllocatorWithDefaultOptions allocator;
        // input/output node info
        std::vector<const char*> input_node_names;
        std::vector<std::vector<int64_t>> input_node_shapes;
        std::vector<const char*> output_node_names;
        std::vector<std::vector<int64_t>> output_node_shapes;
        // model embedding dims
        int32_t embedding_dims = 0;
    };

    enum BackendType {
        TRT = 0,
        ONNX = 1,
    };

  private:
    // model backend type
    BackendType _m_backend_type = TRT;
    // onnx net params
    ONNXParams _m_onnx_params;
    // tokenizer
    std::unique_ptr<TokenizerPtr> _m_tokenizer;
    // init flag
    bool _m_successfully_initialized = false;

  private:
    /***
     *
     * @param config
     * @return
     */
    StatusCode init_onnx(const toml::value& config);

    /***
     *
     * @param in
     * @param out
     * @return
     */
    StatusCode onnx_run(const INPUT& in, OUTPUT& out);
};

/***
*
* @param cfg_file_path
* @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode JinaEmbeddingsV3<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse("")) &config) {
    // choose backend type
    auto backend_dict = config.at("BACKEND_DICT");
    auto backend_name = config.at("JINA_EMBEDDING_V3").at("backend_type").as_string();
    _m_backend_type = static_cast<BackendType>(backend_dict[backend_name].as_integer());

    // init tokenizer
    auto tokenizer_cfg = config.at("TOKENIZER");
    std::string tokenizer_cfg_path = tokenizer_cfg["model_file_path"].as_string();
    if (!FilePathUtil::is_file_exist(tokenizer_cfg_path)) {
        LOG(ERROR) << fmt::format("tokenizer cfg path: {} not exist", tokenizer_cfg_path);
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    auto token_cfg = toml::parse(tokenizer_cfg_path);
    _m_tokenizer = std::make_unique<TokenizerPtr>();
    _m_tokenizer->init(token_cfg);
    if (!_m_tokenizer->is_successfully_initialized()) {
        LOG(ERROR) << "init tokenizer failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init jina embedding configs
    toml::value embedding_cfg;
    if (_m_backend_type == TRT) {
        embedding_cfg = config.at("JINA_EMBEDDING_V3_TRT");
    } else if (_m_backend_type == ONNX) {
        embedding_cfg = config.at("JINA_EMBEDDING_V3_ONNX");
    } else {
        LOG(ERROR) << fmt::format("unsupported backend type: {}", static_cast<int>(_m_backend_type));
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    std::string model_file_name = FilePathUtil::get_file_name(embedding_cfg.at("model_file_path").as_string());

    StatusCode init_status;
    if (_m_backend_type == TRT) {
        LOG(ERROR) << "trt backend has not been implemented";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    } else {
        init_status = init_onnx(embedding_cfg);
    }

    if (init_status == StatusCode::OK) {
        _m_successfully_initialized = true;
        LOG(INFO) << fmt::format("Successfully load jina embedding v3 model from: {}", model_file_name);
    } else {
        _m_successfully_initialized = false;
        LOG(INFO) << fmt::format("Failed load jina embedding v3 model from: {}", model_file_name);
    }

    return init_status;
}

/***
 *
 * @param in
 * @param out
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode JinaEmbeddingsV3<INPUT, OUTPUT>::Impl::run(const INPUT& in, OUTPUT& out) {
    StatusCode infer_status;
    if (_m_backend_type == TRT) {
        LOG(ERROR) << "trt backend has not been implemented";
        infer_status = StatusCode::MODEL_RUN_SESSION_FAILED;
    } else {
        infer_status = onnx_run(in, out);
    }

    return infer_status;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param config
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode JinaEmbeddingsV3<INPUT, OUTPUT>::Impl::init_onnx(const toml::value &config) {
    // ort env and memo info
    _m_onnx_params.env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "");

    // init light glue session
    _m_onnx_params.model_file_path = config.at("model_file_path").as_string();
    if (!FilePathUtil::is_file_exist(_m_onnx_params.model_file_path)) {
        LOG(ERROR) << "jina embedding v3 model file path: " << _m_onnx_params.model_file_path << " not exists";
        return StatusCode::MODEL_INIT_FAILED;
    }
    bool use_gpu = false;
    _m_onnx_params.device = config.at("compute_backend").as_string();
    if (std::strcmp(_m_onnx_params.device.c_str(), "cuda") == 0) {
        use_gpu = true;
        _m_onnx_params.device_id = static_cast<int>(config.at("gpu_device_id").as_integer());
    }
    _m_onnx_params.thread_nums = config.at("model_threads_num").as_integer();
    _m_onnx_params.session_options = Ort::SessionOptions();
    _m_onnx_params.session_options.SetIntraOpNumThreads(_m_onnx_params.thread_nums);
    _m_onnx_params.session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    _m_onnx_params.session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    if (use_gpu) {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = _m_onnx_params.device_id;
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
        cuda_options.gpu_mem_limit = 0;
        cuda_options.arena_extend_strategy = 1;
        cuda_options.do_copy_in_default_stream = 1;
        cuda_options.has_user_compute_stream = 0;
        cuda_options.default_memory_arena_cfg = nullptr;
        _m_onnx_params.session_options.AppendExecutionProvider_CUDA(cuda_options);
        _m_onnx_params.session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    }
    _m_onnx_params.session = new Ort::Session(_m_onnx_params.env, _m_onnx_params.model_file_path.c_str(), _m_onnx_params.session_options);

    // init input/output nodes info
    auto input_nodes_counts = _m_onnx_params.session->GetInputCount();
    for (size_t i = 0 ; i < input_nodes_counts ; i++) {
        auto input_node_name = strdup(_m_onnx_params.session->GetInputNameAllocated(i, _m_onnx_params.allocator).get());
        auto input_node_shape = _m_onnx_params.session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        _m_onnx_params.input_node_names.push_back(std::move(input_node_name));
        _m_onnx_params.input_node_shapes.push_back(input_node_shape);
    }

    auto output_nodes_counts = _m_onnx_params.session->GetOutputCount();
    for (size_t i = 0 ; i < output_nodes_counts ; i++) {
        auto output_node_name = strdup(_m_onnx_params.session->GetOutputNameAllocated(i, _m_onnx_params.allocator).get());
        auto output_node_shape = _m_onnx_params.session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        _m_onnx_params.output_node_names.push_back(std::move(output_node_name));
        _m_onnx_params.output_node_shapes.push_back(output_node_shape);
        if (std::strcmp(output_node_name, "text_embeds") == 0) {
            _m_onnx_params.embedding_dims = output_node_shape.back();
        }
    }

    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param in
 * @param out
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode JinaEmbeddingsV3<INPUT, OUTPUT>::Impl::onnx_run(const INPUT &in, OUTPUT &out) {
    // init sess
    auto& sess = _m_onnx_params.session;
    auto& input_node_shapes = _m_onnx_params.input_node_shapes;
    auto& input_node_names = _m_onnx_params.input_node_names;
    auto& output_node_shapes = _m_onnx_params.output_node_shapes;
    auto& output_node_names = _m_onnx_params.output_node_names;
    auto& embed_dims = _m_onnx_params.embedding_dims;
    jina_embedding_impl::internal_output embed_out;

    // transform external input into internal input
    auto input = jina_embedding_impl::transform_input(in);
    std::string input_text = input.text;
    auto pooling_type = input.pooling_type;

    // tokenize text
    auto status = _m_tokenizer->tokenize(input_text, embed_out.token_ids);
    if (status != StatusCode::OK) {
        LOG(ERROR) << fmt::format("tokenizing input text: {} failed, status code: {}", input_text, status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    std::vector<int64_t > input_tokens_64;
    for (auto& id : embed_out.token_ids) {
        input_tokens_64.push_back(static_cast<int64_t >(id));
    }
    std::vector<int64_t > attn_mask(embed_out.token_ids.size(), 1);
    std::vector<int64_t > task_id = {4};

    // prepare input tensors
    std::vector<Ort::Value> input_tensors;
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault);

    std::vector<int64_t > input_ids_shapes = {1, static_cast<int64_t >(embed_out.token_ids.size())};
    auto input_ids_tensor = Ort::Value::CreateTensor<int64_t >(
        memory_info, input_tokens_64.data(), embed_out.token_ids.size(),
        input_ids_shapes.data(), input_ids_shapes.size());
    input_tensors.push_back(std::move(input_ids_tensor));

    std::vector<int64_t > attn_mask_shapes = {1, static_cast<int64_t >(attn_mask.size())};
    auto attn_mask_tensor = Ort::Value::CreateTensor<int64_t >(
        memory_info, attn_mask.data(), attn_mask.size(),
        attn_mask_shapes.data(), attn_mask_shapes.size());
    input_tensors.push_back(std::move(attn_mask_tensor));

    std::vector<int64_t > task_id_shapes = {1};
    auto task_id_tensor = Ort::Value::CreateTensor<int64_t >(
        memory_info, task_id.data(), task_id.size(),
        task_id_shapes.data(), task_id_shapes.size());
    input_tensors.push_back(std::move(task_id_tensor));

    // run session
    auto output_tensors = sess->Run(
        Ort::RunOptions{nullptr}, input_node_names.data(), input_tensors.data(), input_tensors.size(),
        output_node_names.data() , output_node_names.size());

    // copy output tensor values
    auto& out_embedding_tensor = output_tensors[0];
    int out_embeds_count = 1;
    for (auto &val : out_embedding_tensor.GetTensorTypeAndShapeInfo().GetShape()) {
        out_embeds_count *= val;
    }
    auto token_nums = input_tokens_64.size();
    assert(token_nums * embed_dims == out_embeds_count);
    embed_out.token_embeds.resize(token_nums, std::vector<float>(embed_dims, 0.0));
    for (auto i = 0; i < token_nums; ++i) {
        for (auto j = 0; j < embed_dims; ++j) {
            embed_out.token_embeds[i][j] = out_embedding_tensor.template GetTensorMutableData<float>()[i * embed_dims + j];
        }
    }

    // pool embeddings if needed
    if (pooling_type == pool_type::EMBEDDING_MEAN_POOLING) {
        for (auto col = 0; col < embed_dims; ++col) {
            float sum = 0.0f;
            for (auto row = 0; row < token_nums; ++row) {
                sum += embed_out.token_embeds[row][col];
            }
            embed_out.token_embeds[0][col] = sum / static_cast<float>(token_nums);
        }
    }

    // norm output embeddings
    for (auto row = 0; row < token_nums; ++row) {
        float sum = 0.0f;
        for (auto& val : embed_out.token_embeds[row]) {
            sum += static_cast<float>(std::pow(val, 2));
        }
        sum = std::sqrt(sum);
        for (auto& val : embed_out.token_embeds[row]) {
            val /= sum;
        }
    }

    // transform output
    out = jina_embedding_impl::transform_output<OUTPUT>(embed_out);

    return StatusCode::OK;
}

/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT>
JinaEmbeddingsV3<INPUT, OUTPUT>::JinaEmbeddingsV3() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT>
JinaEmbeddingsV3<INPUT, OUTPUT>::~JinaEmbeddingsV3() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode JinaEmbeddingsV3<INPUT, OUTPUT>::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template <typename INPUT, typename OUTPUT>
bool JinaEmbeddingsV3<INPUT, OUTPUT>::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param input
 * @param output
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode JinaEmbeddingsV3<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

}
}
}
}