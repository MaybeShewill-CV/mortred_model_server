/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: nano_detector.h
* Date: 22-6-10
************************************************/

#ifndef MORTRED_MODEL_SERVER_NANO_DETECTOR_H
#define MORTRED_MODEL_SERVER_NANO_DETECTOR_H

#include <memory>

#include "toml/toml.hpp"

#include "models/base_model.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace jinq {
namespace models {
namespace object_detection {

template<typename INPUT, typename OUTPUT>
class NanoDetector : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
    typedef struct CenterPrior_ {
        int x;
        int y;
        int stride;
    } CenterPrior;

public:

    /***
    * constructor
    * @param config
    */
    NanoDetector();

    /***
     *
     */
    ~NanoDetector() override;

    /***
    * constructor
    * @param transformer
    */
    NanoDetector(const NanoDetector& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    NanoDetector& operator=(const NanoDetector& transformer) = delete;

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
     * if nano detector successfully initialized
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

#endif //MORTRED_MODEL_SERVER_NANO_DETECTOR_H
