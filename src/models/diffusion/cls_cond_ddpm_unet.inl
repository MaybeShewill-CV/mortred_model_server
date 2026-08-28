/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: cls_cond_ddpm_unet.inl
 * Date: 26-8-17
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
using jinq::common::StatusCode;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::TensorInfo;

namespace {

const TensorInfo *find_info(const std::vector<TensorInfo> &infos, const std::string &name) {
    for (const auto &info : infos) {
        if (info.name == name) {
            return &info;
        }
    }
    return nullptr;
}

bool make_integer_scalar(NamedTensor &named, const TensorInfo &info, int64_t value) {
    named.tensor = jinq::models::backend::Tensor::make<int64_t>(info.shape);
    named.tensor.data<int64_t>()[0] = value;
    return true;
}

bool make_integer_scalar(NamedTensor &named, const TensorInfo &info, int32_t value) {
    named.tensor = jinq::models::backend::Tensor::make<int32_t>(info.shape);
    named.tensor.data<int32_t>()[0] = value;
    return true;
}

bool write_integer_scalar(NamedTensor &named, const TensorInfo &info, int64_t value) {
    if (info.dtype == jinq::models::backend::DType::I64) {
        return make_integer_scalar(named, info, static_cast<int64_t>(value));
    }
    if (info.dtype == jinq::models::backend::DType::I32) {
        return make_integer_scalar(named, info, static_cast<int32_t>(value));
    }
    return false;
}

} // namespace

template <typename INPUT, typename OUTPUT> jinq::models::PreparedInput ClsCondDDPMUNet<INPUT, OUTPUT>::prepare_inputs(const INPUT &input) {
    const auto &input_infos = this->session().inputs();
    const auto *xt_info = find_info(input_infos, "xt");
    const auto *t_info = find_info(input_infos, "t");
    const auto *cls_info = find_info(input_infos, "cls_id");
    if (xt_info == nullptr || t_info == nullptr || cls_info == nullptr) {
        LOG(ERROR) << "cls cond ddpm unet session does not expose the 'xt'/'t'/'cls_id' inputs";
        return {};
    }
    if (xt_info->dynamic || t_info->dynamic || cls_info->dynamic) {
        LOG(ERROR) << "cls cond ddpm unet inputs must be static";
        return {};
    }

    jinq::models::PreparedInput prepared;
    std::vector<NamedTensor> inputs;
    NamedTensor xt;
    xt.name = "xt";
    xt.tensor = jinq::models::backend::Tensor::make<float>(xt_info->shape);
    if (input.xt.size() * sizeof(float) != xt.tensor.byte_size()) {
        LOG(ERROR) << "xt element count " << input.xt.size() << " mismatches session input " << xt_info->to_string();
        return {};
    }
    std::memcpy(xt.tensor.buffer.data(), input.xt.data(), xt.tensor.byte_size());
    inputs.push_back(std::move(xt));

    NamedTensor timestep;
    timestep.name = "t";
    if (!write_integer_scalar(timestep, *t_info, input.timestep)) {
        LOG(ERROR) << "unsupported cls cond ddpm unet timestep dtype: " << t_info->to_string();
        return {};
    }
    inputs.push_back(std::move(timestep));

    NamedTensor cls_id;
    cls_id.name = "cls_id";
    if (!write_integer_scalar(cls_id, *cls_info, input.cls_id)) {
        LOG(ERROR) << "unsupported cls cond ddpm unet cls_id dtype: " << cls_info->to_string();
        return {};
    }
    inputs.push_back(std::move(cls_id));
    prepared.inputs = std::move(inputs);
    return prepared;
}

template <typename INPUT, typename OUTPUT>
StatusCode ClsCondDDPMUNet<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                       const jinq::models::InferenceContext & /*context*/, OUTPUT &output) {
    if (outputs.empty()) {
        LOG(ERROR) << "cls cond ddpm unet output tensor is empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto &tensor = outputs.front().tensor;
    const auto *data = tensor.data<float>();
    ClsCondUnetOutput internal_out;
    internal_out.predict_noise.resize(static_cast<size_t>(tensor.element_count()));
    std::memcpy(internal_out.predict_noise.data(), data, tensor.byte_size());
    output = std::move(internal_out);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT>
ClsCondDDPMUNet<INPUT, OUTPUT>::ClsCondDDPMUNet() : jinq::models::BackendCvModel<INPUT, OUTPUT>("DDPM_UNET") {}

} // namespace diffusion
} // namespace models
} // namespace jinq
