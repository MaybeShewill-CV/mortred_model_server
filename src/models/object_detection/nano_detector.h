/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: nano_detector.h
* Date: 22-6-10
************************************************/

#ifndef MMAISERVER_NANODETECTOR_H
#define MMAISERVER_NANODETECTOR_H

#include <memory>

#include "toml/toml.hpp"

#include "models/base_model.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace morted {
namespace models {
namespace object_detection {

template<typename INPUT, typename OUTPUT>
class NanoDetector : public morted::models::BaseAiModel<INPUT, OUTPUT> {
public:

    /***
    * 构造函数
    * @param config
    */
    NanoDetector();

    /***
     *
     */
    ~NanoDetector() override;

    /***
    * 赋值构造函数
    * @param transformer
    */
    NanoDetector(const NanoDetector& transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    NanoDetector& operator=(const NanoDetector& transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    morted::common::StatusCode init(const decltype(toml::parse(""))& cfg) override;

    /***
     *
     * @param input
     * @param output
     * @return
     */
    morted::common::StatusCode run(const INPUT& input, std::vector<OUTPUT>& output) override;


    /***
     * if db text detector successfully initialized
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

#include "nano_detector.inl"

#endif //MMAISERVER_NANODETECTOR_H
