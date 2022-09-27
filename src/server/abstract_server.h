/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: base_server.h
* Date: 22-6-21
************************************************/

#ifndef MM_AI_SERVER_BASESERVER_H
#define MM_AI_SERVER_BASESERVER_H

#include <toml/toml.hpp>
#include <workflow/WFTask.h>
#include <workflow/WFHttpServer.h>

#include "common/status_code.h"

namespace jinq {
namespace server {
class BaseAiServer {
public:
    /***
    *
    */
    virtual ~BaseAiServer() = default;

    /***
     * 构造函数
     * @param config
     */
    BaseAiServer() = default;

    /***
     *
     * @param cfg
     * @return
     */
    virtual jinq::common::StatusCode init(const decltype(toml::parse(""))& cfg) = 0;

    /***
     *
     * @param input
     * @param output
     * @return
     */
    virtual void serve_process(WFHttpTask* task) = 0;

    /***
     *
     * @return
     */
    virtual bool is_successfully_initialized() const = 0;

    /***
 *
 * @param port
 * @return
 */
    inline int start(unsigned short port) {
        return _m_server->start(port);
    };

    /***
     *
     * @param host
     * @param port
     * @return
     */
    inline int start(const char *host, unsigned short port) {
        return _m_server->start(host, port);
    };

    /***
     *
     */
    inline void stop() {
        return _m_server->stop();
    };

    /***
     *
     */
    inline void shutdown() {
        _m_server->shutdown();
    };

    /***
     *
     */
    inline void wait_finish() {
        _m_server->wait_finish();
    }

protected:
    std::unique_ptr<WFHttpServer> _m_server;
};
}
}

#endif //MM_AI_SERVER_BASESERVER_H
