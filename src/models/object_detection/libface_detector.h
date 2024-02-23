/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: libface_detector.h
 * Date: 22-6-10
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_LIBFACE_DETECTOR_H
#define MORTRED_MODEL_SERVER_LIBFACE_DETECTOR_H

#include <memory>

#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/base_model.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace object_detection {

template <typename INPUT, typename OUTPUT> 
class LibFaceDetector : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
  public:
    /***
     * constructor
     * @param config
     */
    LibFaceDetector();

    /***
     *
     */
    ~LibFaceDetector() override;

    /***
     * constructor
     * @param transformer
     */
    LibFaceDetector(const LibFaceDetector &transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    LibFaceDetector &operator=(const LibFaceDetector &transformer) = delete;

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

#include "libface_detector.inl"

#endif // MORTRED_MODEL_SERVER_LIBFACE_DETECTOR_H
