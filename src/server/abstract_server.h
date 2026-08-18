/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: base_server.h
* Date: 22-6-21
************************************************/

#ifndef MORTRED_MODEL_SERVER_BASESERVER_H
#define MORTRED_MODEL_SERVER_BASESERVER_H

#include <memory>

#include <toml/toml.hpp>
#include <workflow/WFTask.h>
#include <workflow/WFHttpServer.h>
#include <workflow/Workflow.h>

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
    virtual jinq::common::StatusCode init(const toml::table& cfg) = 0;

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
    /***
     * 通用 HTTP 服务装配：workflow 全局设置 + WFServerParams + WFHttpServer。
     * 参数全部取自 Impl 的 _m_* 成员（基类有默认值兜底），
     * 特异 server 可以不调用本方法而自行装配 _m_server。
     */
    template<typename IMPL>
    jinq::common::StatusCode init_http_server(IMPL* impl) {
        WFGlobalSettings settings = GLOBAL_SETTINGS_DEFAULT;
        settings.compute_threads = impl->_m_compute_threads;
        settings.handler_threads = impl->_m_handler_threads;
        WORKFLOW_library_init(&settings);

        WFServerParams server_params = SERVER_PARAMS_DEFAULT;
        server_params.max_connections = impl->_m_max_connection_nums;
        server_params.peer_response_timeout = impl->_m_peer_resp_timeout;
        server_params.request_size_limit = impl->_m_request_size_limit * 1024 * 1024;

        auto&& proc = [impl](auto arg) { return impl->serve_process(arg); };
        _m_server = std::make_unique<WFHttpServer>(&server_params, proc);
        return jinq::common::StatusCode::OK;
    }

    std::unique_ptr<WFHttpServer> _m_server;
};
}
}

#endif //MORTRED_MODEL_SERVER_BASESERVER_H
