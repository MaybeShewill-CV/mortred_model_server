/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: DBTextDetector.h
* Date: 22-6-6
************************************************/

#ifndef MMAISERVER_DBTextDetector_H
#define MMAISERVER_DBTextDetector_H

#include <memory>

#include "toml/toml.hpp"

#include "models/base_model.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace morted {
namespace models {
namespace image_ocr {

template<typename INPUT, typename OUTPUT>
class DBTextDetector : public morted::models::BaseAiModel<INPUT, OUTPUT> {
public:

    /***
    * 构造函数
    * @param config
    */
    DBTextDetector();

    /***
     *
     */
    ~DBTextDetector() override;

    /***
    * 赋值构造函数
    * @param transformer
    */
    DBTextDetector(const DBTextDetector& transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    DBTextDetector& operator=(const DBTextDetector& transformer) = delete;

    /***
    * get db text detector instance
    * @return lp detector single instance
    */
    static DBTextDetector& get_instance() {
        static DBTextDetector instance;
        return instance;
    }

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

};
}
}

#include "models/image_ocr/db_text_detector.inl"

#endif //MMAISERVER_DBTextDetector_H
