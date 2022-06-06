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
#include "common/status_code.h"

namespace morted {
namespace models {
namespace image_ocr {

struct dbtext_input {

};

struct dbtext_output {

};

class DBTextDetector : public morted::models::BaseAiModel {
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
    template<class INPUT, class OUTPUT>
    typename std::enable_if <
    std::is_same<INPUT, dbtext_input>::value&& std::is_same<OUTPUT, dbtext_output>::value,
        morted::common::StatusCode >::type
        run(const INPUT* input, OUTPUT* output);

    template<class INPUT, class OUTPUT>
    typename std::enable_if < !std::is_same<INPUT, dbtext_input>::value, morted::common::StatusCode >::type
    run(const INPUT* input, OUTPUT* output);

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
