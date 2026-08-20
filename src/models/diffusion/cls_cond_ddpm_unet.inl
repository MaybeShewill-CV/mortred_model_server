/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: cls_cond_ddpm_unet.cpp
 ************************************************/

#include "cls_cond_ddpm_unet.h"

#include <algorithm>
#include <cstring>

#include "glog/logging.h"

namespace jinq {
namespace models {
namespace diffusion {

using ClsCondUnetInput = jinq::models::io_define::diffusion::std_cls_cond_ddpm_unet_input;
using ClsCondUnetOutput = jinq::models::io_define::diffusion::std_cls_cond_ddpm_unet_output;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::TensorInfo;
using jinq::common::StatusCode;

namespace {

const TensorInfo* find_info(const std::vector<TensorInfo>& infos, const std::string& name) {
    for (const auto& info : infos) {
        if (info.name == name) {
            return &info;
        }
    }
    return nullptr;
}

}  // namespace

template <typename INPUT, typename OUTPUT>
std::vector<NamedTensor> ClsCondDDPMUNet<INPUT, OUTPUT>::make_inputs(const INPUT& input) {
    const auto& input_infos = this->session().inputs();
    const auto* xt_info = find_info(input_infos, "xt");
    const auto* t_info = find_info(input_infos, "t");
    const auto* cls_info = find_info(input_infos, "cls_id");
    if (xt_info == nullptr || t_info == nullptr || cls_info == nullptr) {
        LOG(ERROR) << "cls cond ddpm unet session does not expose the 'xt'/'t'/'cls_id' inputs";
        return {};
    }
    if (xt_info->dynamic || t_info->dynamic || cls_info->dynamic) {
        LOG(ERROR) << "cls cond ddpm unet inputs must be static";
        return {};
    }

    std::vector<NamedTensor> inputs;
    NamedTensor xt;
    xt.name = "xt";
    xt.tensor = jinq::models::backend::Tensor::make<float>(xt_info->shape);
    if (input.xt.size() * sizeof(float) != xt.tensor.byte_size()) {
        LOG(ERROR) << "xt element count " << input.xt.size() << " mismatches session input "
                   << xt_info->to_string();
        return {};
    }
    std::memcpy(xt.tensor.buffer.data(), input.xt.data(), xt.tensor.byte_size());
    inputs.push_back(std::move(xt));

    NamedTensor timestep;
    timestep.name = "t";
    timestep.tensor = jinq::models::backend::Tensor::make<int64_t>(t_info->shape);
    timestep.tensor.data<int64_t>()[0] = input.timestep;
    inputs.push_back(std::move(timestep));

    NamedTensor cls_id;
    cls_id.name = "cls_id";
    cls_id.tensor = jinq::models::backend::Tensor::make<int64_t>(cls_info->shape);
    cls_id.tensor.data<int64_t>()[0] = input.cls_id;
    inputs.push_back(std::move(cls_id));
    return inputs;
}

template <typename INPUT, typename OUTPUT>
StatusCode ClsCondDDPMUNet<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor>& outputs,
                                                       OUTPUT& output) {
    if (outputs.empty()) {
        LOG(ERROR) << "cls cond ddpm unet output tensor is empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto& tensor = outputs.front().tensor;
    const auto* data = tensor.data<float>();
    ClsCondUnetOutput internal_out;
    internal_out.predict_noise.resize(static_cast<size_t>(tensor.element_count()));
    std::memcpy(internal_out.predict_noise.data(), data, tensor.byte_size());
    output = std::move(internal_out);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT>
ClsCondDDPMUNet<INPUT, OUTPUT>::ClsCondDDPMUNet()
    : jinq::models::BackendCvModel<INPUT, OUTPUT>("DDPM_UNET") {}

} // namespace diffusion
} // namespace models
} // namespace jinq
