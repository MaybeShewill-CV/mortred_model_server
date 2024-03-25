/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: dbnet_server.h
* Date: 22-7-04
************************************************/

#ifndef MORTRED_MODEL_SERVER_DBNET_SERVER_H
#define MORTRED_MODEL_SERVER_DBNET_SERVER_H

#include <memory>

#include "server/abstract_server.h"

namespace jinq {
namespace server {
namespace ocr {
class DBNetServer : public jinq::server::BaseAiServer {
public:

    /***
    * constructor
    * @param config
    */
    DBNetServer();

    /***
     *
     */
    ~DBNetServer() override;

    /***
    * constructor
    * @param transformer
    */
    DBNetServer(const DBNetServer& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    DBNetServer& operator=(const DBNetServer& transformer) = delete;

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

#endif //MORTRED_MODEL_SERVER_DBNET_SERVER_H
