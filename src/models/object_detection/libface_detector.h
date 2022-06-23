/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: libface_detector.h
 * Date: 22-6-10
 ************************************************/

#ifndef MMAISERVER_LIBFACEDETECTOR_H
#define MMAISERVER_LIBFACEDETECTOR_H

#include <memory>

#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/base_model.h"
#include "models/model_io_define.h"

namespace morted {
namespace models {
namespace object_detection {

template <typename INPUT, typename OUTPUT> 
class LibFaceDetector : public morted::models::BaseAiModel<INPUT, OUTPUT> {
  public:
    /***
     * 构造函数
     * @param config
     */
    LibFaceDetector();

    /***
     *
     */
    ~LibFaceDetector() override;

    /***
     * 赋值构造函数
     * @param transformer
     */
    LibFaceDetector(const LibFaceDetector &transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    LibFaceDetector &operator=(const LibFaceDetector &transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    morted::common::StatusCode init(const decltype(toml::parse("")) &cfg) override;

    /***
     *
     * @param input
     * @param output
     * @return
     */
    morted::common::StatusCode run(const INPUT &input, OUTPUT &output) override;

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
} // namespace morted

#include "libface_detector.inl"

#endif // MMAISERVER_LIBFACEDETECTOR_H
