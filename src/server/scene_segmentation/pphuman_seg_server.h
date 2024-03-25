/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: pphuman_seg_server.h
* Date: 22-7-22
************************************************/

#ifndef MORTRED_MODEL_SERVER_PPHUMAN_SEG_SERVER_H
#define MORTRED_MODEL_SERVER_PPHUMAN_SEG_SERVER_H

#include <memory>

#include "server/abstract_server.h"

namespace jinq {
namespace server {
namespace scene_segmentation {
class PPHumanSegServer : public jinq::server::BaseAiServer {
public:

    /***
    * constructor
    * @param config
    */
    PPHumanSegServer();

    /***
     *
     */
    ~PPHumanSegServer() override;

    /***
    * constructor
    * @param transformer
    */
    PPHumanSegServer(const PPHumanSegServer& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    PPHumanSegServer& operator=(const PPHumanSegServer& transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    jinq::common::StatusCode init(const decltype(toml::parse(""))& cfg) override;

    /***
     *
     * @param task
     */
    void serve_process(WFHttpTask* task) override;

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const override;

private:
    class Impl;
    std::unique_ptr<Impl> _m_impl;
};
}
}
}

#endif //MORTRED_MODEL_SERVER_PPHUMAN_SEG_SERVER_H
