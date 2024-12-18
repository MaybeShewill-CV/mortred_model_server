/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: jina_embeddings_v3.h
 * Date: 24-12-17
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_JINA_EMBEDDINGS_V3_H
#define MORTRED_MODEL_SERVER_JINA_EMBEDDINGS_V3_H

#include <memory>

#include "toml/toml.hpp"

#include "models/base_model.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace jinq {
namespace models {
namespace llm {
namespace embedding {

template <typename INPUT, typename OUTPUT> 
class JinaEmbeddingsV3 : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
  public:
    /***
    * constructor
    * @param config
     */
    JinaEmbeddingsV3();

    /***
     *
     */
    ~JinaEmbeddingsV3() override;

    /***
    * constructor
    * @param transformer
     */
    JinaEmbeddingsV3(const JinaEmbeddingsV3 &transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    JinaEmbeddingsV3 &operator=(const JinaEmbeddingsV3 &transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    jinq::common::StatusCode init(const decltype(toml::parse("")) &cfg) override;

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

#include "jina_embeddings_v3.inl"

#endif // MORTRED_MODEL_SERVER_JINA_EMBEDDINGS_V3_H
