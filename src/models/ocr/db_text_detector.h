/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: db_text_detector.h
* Date: 22-6-6
************************************************/

#ifndef MORTRED_MODEL_SERVER_DB_TEXT_DETECTOR_H
#define MORTRED_MODEL_SERVER_DB_TEXT_DETECTOR_H

#include <memory>

#include "toml/toml.hpp"

#include "models/base_model.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace jinq {
namespace models {
namespace ocr {

template<typename INPUT, typename OUTPUT>
class DBTextDetector : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
public:

    /***
    * constructor
    * @param config
    */
    DBTextDetector();

    /***
     *
     */
    ~DBTextDetector() override;

    /***
    * constructor
    * @param transformer
    */
    DBTextDetector(const DBTextDetector& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    DBTextDetector& operator=(const DBTextDetector& transformer) = delete;

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

#include "db_text_detector.inl"

#endif //MORTRED_MODEL_SERVER_DB_TEXT_DETECTOR_H
