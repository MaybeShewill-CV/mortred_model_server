/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: hrnet_segmentation.h
 * Date: 23-11-17
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_HRNETSEGMENTATION_H
#define MORTRED_MODEL_SERVER_HRNETSEGMENTATION_H

#include <memory>

#include "toml/toml.hpp"

#include "models/base_model.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace jinq {
namespace models {
namespace scene_segmentation {

template<typename INPUT, typename OUTPUT>
class HRNetSegmentation : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
  public:

    /***
    * constructor
    * @param config
     */
    HRNetSegmentation();

    /***
     *
     */
    ~HRNetSegmentation() override;

    /***
    * constructor
    * @param transformer
     */
    HRNetSegmentation(const HRNetSegmentation& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    HRNetSegmentation& operator=(const HRNetSegmentation& transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    jinq::common::StatusCode init(const decltype(toml::parse(""))& cfg) override;

    /***
     *
     * @param input
     * @param output
     * @return
     */
    jinq::common::StatusCode run(const INPUT& input, OUTPUT& output) override;


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

#include "hrnet_segmentation.inl"

#endif // MORTRED_MODEL_SERVER_HRNETSEGMENTATION_H
