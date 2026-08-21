/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: ddpm_unet.inl
 * Date: 24-4-23
 ************************************************/

#include "ddpm_unet.h"

#include <algorithm>
#include <cstring>

#include "glog/logging.h"

namespace jinq {
namespace models {
namespace diffusion {

using UnetInput = jinq::models::io_define::diffusion::std_ddpm_unet_input;
using UnetOutput = jinq::models::io_define::diffusion::std_ddpm_unet_output;
using jinq::models::backend::NamedTensor;
using jinq::common::StatusCode;

template <typename INPUT, typename OUTPUT>
std::vector<NamedTensor> DDPMUNet<INPUT, OUTPUT>::make_inputs(const INPUT& input) {
    const auto& input_infos = this->session().inputs();
    const auto xt_info = std::find_if(
        input_infos.begin(), input_infos.end(),
        [](const jinq::models::backend::TensorInfo& info) { return info.name == "xt"; });
    const auto t_info = std::find_if(
        input_infos.begin(), input_infos.end(),
        [](const jinq::models::backend::TensorInfo& info) { return info.name == "t"; });
    if (xt_info == input_infos.end() || t_info == input_infos.end()) {
        LOG(ERROR) << "ddpm unet session does not expose the 'xt'/'t' inputs";
        return {};
    }
    if (xt_info->dynamic || t_info->dynamic) {
        LOG(ERROR) << "ddpm unet inputs must be static, got " << xt_info->to_string() << " / "
                   << t_info->to_string();
        return {};
    }

    std::vector<NamedTensor> inputs;
    NamedTensor xt;
    xt.name = "xt";
    xt.tensor = jinq::models::backend::Tensor::make<float>(xt_info->shape);
    if (input.xt.size() * sizeof(float) != xt.tensor.byte_size()) {
        LOG(ERROR) << "ddpm unet xt element count " << input.xt.size()
                   << " mismatches session input " << xt_info->to_string();
        return {};
    }
    std::memcpy(xt.tensor.buffer.data(), input.xt.data(), xt.tensor.byte_size());
    inputs.push_back(std::move(xt));

    NamedTensor timestep;
    timestep.name = "t";
    if (t_info->dtype == jinq::models::backend::DType::I64) {
        timestep.tensor = jinq::models::backend::Tensor::make<int64_t>(t_info->shape);
        timestep.tensor.data<int64_t>()[0] = input.timestep;
    } else if (t_info->dtype == jinq::models::backend::DType::I32) {
        // TensorRT builders commonly lower scalar int64 timestep inputs to
        // int32; follow the dtype exposed by the deployed engine.
        timestep.tensor = jinq::models::backend::Tensor::make<int32_t>(t_info->shape);
        timestep.tensor.data<int32_t>()[0] = static_cast<int32_t>(input.timestep);
    } else {
        LOG(ERROR) << "unsupported ddpm unet timestep dtype: " << t_info->to_string();
        return {};
    }
    inputs.push_back(std::move(timestep));
    return inputs;
}

template <typename INPUT, typename OUTPUT>
StatusCode DDPMUNet<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor>& outputs,
                                                OUTPUT& output) {
    if (outputs.empty()) {
        LOG(ERROR) << "ddpm unet output tensor is empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto& tensor = outputs.front().tensor;
    const auto* data = tensor.data<float>();
    UnetOutput internal_out;
    internal_out.predict_noise.resize(static_cast<size_t>(tensor.element_count()));
    std::memcpy(internal_out.predict_noise.data(), data, tensor.byte_size());
    output = std::move(internal_out);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT>
DDPMUNet<INPUT, OUTPUT>::DDPMUNet()
    : jinq::models::BackendCvModel<INPUT, OUTPUT>("DDPM_UNET") {}

} // namespace diffusion
} // namespace models
}
