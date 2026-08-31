/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: base_model.h
* Date: 22-6-2
************************************************/

#ifndef MORTRED_MODEL_SERVER_BASE_MODEL_H
#define MORTRED_MODEL_SERVER_BASE_MODEL_H

#include "toml/toml.hpp"

#include "glog/logging.h"

#include <vector>

#include "common/status_code.h"

namespace jinq {
namespace models {
using jinq::common::StatusCode;

template<typename INPUT, typename OUTPUT>
class BaseAiModel {
public:
    /*** the served input/output contracts; the generic server core asks the
     * worker for its input type instead of hardcoding one (tests mount
     * legacy base64 workers, catalogs mount the unified image_input) */
    using input_type = INPUT;
    using output_type = OUTPUT;

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
    virtual StatusCode init(const toml::table& cfg) = 0;

    /***
     * Non-virtual lifecycle guard (NVI): running an uninitialized model
     * (never inited, or init failed) returns MODEL_INIT_FAILED instead of
     * dereferencing null backend resources. Concrete inference lives in
     * run_impl.
     *
     * @param input
     * @param output
     * @return
     */
    StatusCode run(const INPUT& in, OUTPUT& out) {
        if (!is_successfully_initialized()) {
            LOG(ERROR) << "model is not successfully initialized, refuse to run";
            return StatusCode::MODEL_INIT_FAILED;
        }
        return run_impl(in, out);
    }

    /***
     * Batched inference entry (NVI guarded like run()). item_status is
     * index-aligned with in; the aggregate return is OK only when every item
     * succeeded. Per-item failures (preprocess / postprocess) are ISOLATED:
     * one bad item never fails its batch mates. Session-level failures cannot
     * be attributed to an item and are broadcast to all participating items -
     * implementations must follow this contract. The default loops run_impl
     * per item so every model stays correct under a batch scheduler; batch
     * capable models override it with a single N-dim session run.
     */
    virtual StatusCode run_batch(const std::vector<INPUT>& in,
                                 std::vector<OUTPUT>& out,
                                 std::vector<StatusCode>& item_status) {
        out.assign(in.size(), OUTPUT{});
        item_status.assign(in.size(), StatusCode::OK);
        StatusCode aggregate = StatusCode::OK;
        for (size_t idx = 0; idx < in.size(); ++idx) {
            item_status[idx] = run(in[idx], out[idx]);
            if (item_status[idx] != StatusCode::OK) {
                aggregate = item_status[idx];
            }
        }
        return aggregate;
    }

    /***
     *
     * @param input
     * @param output
     * @return
     */
  protected:
    virtual StatusCode run_impl(const INPUT& in, OUTPUT& out) = 0;

  public:
    /***
     *
     * @return
     */
    virtual bool is_successfully_initialized() const = 0;
};
}
}


#endif //MORTRED_MODEL_SERVER_BASE_MODEL_H
