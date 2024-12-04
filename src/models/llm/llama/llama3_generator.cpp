/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: llama3_generator.cpp
 * Date: 24-11-28
 ************************************************/

#include "llama3_generator.h"

#include "glog/logging.h"

#include "common/cv_utils.h"
#include "common/time_stamp.h"
#include "common/file_path_util.h"
#include "models/llm/llama/llama3.h"
#include "models/llm/chat_template/llama3_chat_template.h"

namespace jinq {
namespace models {
namespace llm {

using jinq::common::CvUtils;
using jinq::common::Timestamp;
using jinq::common::StatusCode;
using jinq::common::FilePathUtil;
using jinq::models::llm::chat_template::Dialog;
using jinq::models::llm::chat_template::ChatMessage;
using jinq::models::llm::chat_template::Llama3ChatTemplate;
using Llama3Ptr = jinq::models::llm::llama::Llama3<std::vector<llama_token>&, std::string>;

namespace llama {

/***************** Impl Function Sets ******************/

class Llama3Generator::Impl {
  public:
    /***
     *
     */
    Impl() = default;

    /***
     *
     */
    ~Impl() = default;

    /***
    *
    * @param transformer
     */
    Impl(const Impl& transformer) = delete;

    /***
     *
     * @param transformer
     * @return
     */
    Impl& operator=(const Impl& transformer) = delete;

    /***
     *
     * @param cfg_file_path
     * @return
     */
    StatusCode init(const decltype(toml::parse("")) &config);

    /***
     *
     * @param input
     * @param output
     * @return
     */
    StatusCode text_completion(const std::string& prompt, OUT std::string& generate_output);

    /***
     *
     * @param dialog
     * @param generate_output
     * @param truncate
     * @return
     */
    StatusCode chat_completion(models::llm::chat_template::Dialog& dialog, OUT std::string& generate_output);

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_initialized;
    };

  private:
    // init flag
    bool _m_successfully_initialized = false;
    // llm model
    Llama3Ptr _m_model;
    // chat template
    Llama3ChatTemplate _m_chat_template;
};

/***
 *
 * @param config
 * @return
 */
StatusCode Llama3Generator::Impl::init(const decltype(toml::parse("")) &config) {
    // init llama3 model
    auto status = _m_model.init(config);
    if (status != StatusCode::OK) {
        _m_successfully_initialized = false;
    } else {
        _m_successfully_initialized = true;
    }
    return StatusCode::OK;
}

/***
 *
 * @param prompt
 * @param generate_output
 * @return
 */
StatusCode Llama3Generator::Impl::text_completion(const std::string &prompt, std::string &generate_output) {
    // tokenize prompts
    std::vector<llama_token> prompt_tokens;
    auto status = _m_model.tokenize_prompt(prompt, prompt_tokens);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "tokenize dialog failed, status code: " << status;
        return status;
    }

    // chat completion
    status = _m_model.run(prompt_tokens, generate_output);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "llama3 model run failed, status: " << status;
        return status;
    }

    return status;
}

/***
 *
 * @param dialog
 * @param generate_output
 * @param truncate
 * @return
 */
StatusCode Llama3Generator::Impl::chat_completion(Dialog &dialog, std::string &generate_output) {
    // template format dialog
    std::string fmt_prompt;
    auto status = _m_chat_template.apply_chat_template(dialog, fmt_prompt);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "apply chat template for dialog failed, status code: " << status;
        return status;
    }

    // tokenize prompts
    std::vector<llama_token> prompt_tokens;
    status = _m_model.tokenize_prompt(fmt_prompt, prompt_tokens);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "tokenize dialog failed, status code: " << status;
        return status;
    }

    // chat completion
    status = _m_model.run(prompt_tokens, generate_output);

    // log dialog messages
//    for (auto& msg : dialog.messages) {
//        DLOG(INFO) << fmt::format("{}: {}", msg.role, msg.content);
//    }
//    DLOG(INFO) << fmt::format("assistant: {}", generate_output);

    return status;
}

/************* Export Function Sets *************/

/***
 *
 */
Llama3Generator::Llama3Generator() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 */
Llama3Generator::~Llama3Generator() = default;

/***
 *
 * @param cfg
 * @return
 */
StatusCode Llama3Generator::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @return
 */
bool Llama3Generator::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

/***
 *
 * @param prompt
 * @param generate_output
 * @return
 */
StatusCode Llama3Generator::text_completion(const std::string &prompt, std::string &generate_output) {
    return _m_pimpl-> text_completion(prompt, generate_output);
}

/***
 *
 * @param dialog
 * @param generate_output
 * @param truncate
 * @return
 */
StatusCode Llama3Generator::chat_completion(models::llm::chat_template::Dialog &dialog, std::string &generate_output) {
    return _m_pimpl->chat_completion(dialog, generate_output);
}

}
}
}
}