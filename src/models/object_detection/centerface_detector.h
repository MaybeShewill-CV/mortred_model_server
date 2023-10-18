/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: centerface_detector.h
 * Date: 23-10-18
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_CENTERFACE_DETECTOR_H
#define MORTRED_MODEL_SERVER_CENTERFACE_DETECTOR_H

#include <memory>

#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/base_model.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace object_detection {

template <typename INPUT, typename OUTPUT> 
class CenterFaceDetector : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
  public:
    /***
     * constructor
     * @param config
     */
    CenterFaceDetector();

    /***
     *
     */
    ~CenterFaceDetector() override;

    /***
     * constructor
     * @param transformer
     */
    CenterFaceDetector(const CenterFaceDetector &transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    CenterFaceDetector &operator=(const CenterFaceDetector &transformer) = delete;

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
     * if db text detector successfully initialized
     * @return
     */
    bool is_successfully_initialized() const override;

  private:
    class Impl;
    std::unique_ptr<Impl> _m_pimpl;
};
} // namespace object_detection
} // namespace models
} // namespace jinq

#include "centerface_detector.inl"

#endif // MORTRED_MODEL_SERVER_CENTERFACE_DETECTOR_H
