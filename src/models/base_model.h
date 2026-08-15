/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: base_model.h
* Date: 22-6-2
************************************************/

#ifndef MORTRED_MODEL_SERVER_BASE_MODEL_H
#define MORTRED_MODEL_SERVER_BASE_MODEL_H

#include "toml/toml.hpp"

#include "common/status_code.h"

namespace jinq {
namespace models {

template<typename INPUT, typename OUTPUT>
class BaseAiModel {
public:
    /***
    *
    */
    virtual ~BaseAiModel() = default;

    /***
     * 
     * @param config
     */
    BaseAiModel() = default;

    // polymorphic base: copy deleted to prevent slicing (C++ Core Guidelines C.67)
    BaseAiModel(const BaseAiModel& transformer) = delete;

    /***
     *
     * @param transformer
     * @return
     */
    BaseAiModel& operator=(const BaseAiModel& transformer) = delete;

    /***
     *
     * @param cfg
     * @return
     */
    virtual jinq::common::StatusCode init(const toml::table& cfg) = 0;

    /***
     *
     * @param input
     * @param output
     * @return
     */
    virtual jinq::common::StatusCode run(const INPUT& in, OUTPUT& out) = 0;

    /***
     *
     * @return
     */
    virtual bool is_successfully_initialized() const = 0;
};
}
}


#endif //MORTRED_MODEL_SERVER_BASE_MODEL_H
