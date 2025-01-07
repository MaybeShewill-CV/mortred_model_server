/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: qwen2_vl.h
 * Date: 25-1-6
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_QWEN2_VL_H
#define MORTRED_MODEL_SERVER_QWEN2_VL_H

#include <memory>

#include "toml/toml.hpp"
#include "llama_cpp/llama.h"

#include "common/status_code.h"
#include "models/base_model.h"
#include "models/model_io_define.h"
#include "models/llm/llm_datatype.hpp"

namespace jinq {
namespace models {
namespace llm {

namespace qwen2_vl {

template <typename INPUT, typename OUTPUT>
class Qwen2VL : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
  public:
    /***
    * constructor
    * @param config
     */
    Qwen2VL();

    /***
     *
     */
    ~Qwen2VL() override;

    /***
    * constructor
    * @param transformer
     */
    Qwen2VL(const Qwen2VL &transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    Qwen2VL &operator=(const Qwen2VL &transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    jinq::common::StatusCode init(const toml::value& cfg) override;

    /***
     *
     * @param input
     * @param output
     * @return
     */
    jinq::common::StatusCode run(const INPUT &input, OUTPUT &output) override;

    /***
     * if model successfully initialized
     * @return
     */
    bool is_successfully_initialized() const override;

  private:
    class Impl;
    std::unique_ptr<Impl> _m_pimpl;
};

}
}
}
}

#include "qwen2_vl.inl"

#endif // MORTRED_MODEL_SERVER_QWEN2_VL_H
