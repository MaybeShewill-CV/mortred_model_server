/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: Llama3ChatTemplate_chat_template.inl
 * Date: 24-11-26
 ************************************************/

#include "models/llm/chat_template/llama3_chat_template.h"

#include "fmt/format.h"

namespace jinq {
namespace models {
namespace llm {

using jinq::common::StatusCode;

namespace chat_template {

class Llama3ChatTemplate::Impl {
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
     * @param dialog
     * @param out_fmt_str
     * @return
     */
    StatusCode apply_chat_template(const Dialog& dialog, std::string& out_fmt_str);

  private:
    std::string _m_header_fmt = "<|start_header_id|>{}<|end_header_id|>\n\n";
    std::string _m_message_fmt = "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>";

};

/***
 *
 * @param messages
 * @param out_fmt_str
 * @return
 */
StatusCode Llama3ChatTemplate::Impl::apply_chat_template(const Dialog& dialog, std::string &out_fmt_str) {
    if (dialog.empty()) {
        return StatusCode::TOKENIZE_FAILED;
    }
    std::string fmt_dialog;
    for (auto& message : dialog.messages) {
        fmt_dialog += fmt::format(_m_message_fmt, message.role, message.content);
    }
    fmt_dialog += fmt::format(_m_header_fmt, "assistant");
    out_fmt_str = fmt_dialog;
    return StatusCode::OK;
}


/************* Export Function Sets *************/

/***
 *
 */
Llama3ChatTemplate::Llama3ChatTemplate() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 */
Llama3ChatTemplate::~Llama3ChatTemplate() = default;

/***
 *
 * @param messages
 * @param out_fmt_str
 * @return
 */
StatusCode Llama3ChatTemplate::apply_chat_template(const Dialog& dialog, std::string &out_fmt_str) {
    return _m_pimpl->apply_chat_template(dialog, out_fmt_str);
}

}
}
}
}
