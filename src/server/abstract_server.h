/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: abstract_server.h
* Date: 22-6-21
************************************************/

#ifndef MORTRED_MODEL_SERVER_BASESERVER_H
#define MORTRED_MODEL_SERVER_BASESERVER_H

#include <memory>
#include <mutex>

#include <toml/toml.hpp>
#include <workflow/WFTask.h>
#include <workflow/WFHttpServer.h>
#include <workflow/Workflow.h>

#include "common/status_code.h"

namespace jinq {
namespace server {
using jinq::common::StatusCode;
class BaseAiServer {
public:
    /***
    *
    */
    virtual ~BaseAiServer() = default;

    /***
     * Constructor
     * @param config
     */
    BaseAiServer() = default;

    /***
     *
     * @param cfg
     * @return
     */
    virtual StatusCode init(const toml::table& cfg) = 0;

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
     * Generic HTTP server assembly: workflow global settings + WFServerParams + WFHttpServer.
     * All params come from the Impl's _m_* members (base class provides defaults);
     * specific servers may skip this and assemble _m_server themselves.
     */
    template<typename IMPL>
    StatusCode init_http_server(IMPL* impl) {
        // workflow global settings may be initialized only once: with multiple servers
        // in one process (tests/gateway), the first init wins to avoid re-init UB.
        static std::once_flag workflow_init_flag;
        std::call_once(workflow_init_flag, [impl]() {
            WFGlobalSettings settings = GLOBAL_SETTINGS_DEFAULT;
            settings.compute_threads = impl->_m_compute_threads;
            settings.handler_threads = impl->_m_handler_threads;
            WORKFLOW_library_init(&settings);
        });

        WFServerParams server_params = SERVER_PARAMS_DEFAULT;
        server_params.max_connections = impl->_m_max_connection_nums;
        server_params.peer_response_timeout = impl->_m_peer_resp_timeout;
        server_params.request_size_limit = impl->_m_request_size_limit * 1024 * 1024;

        auto&& proc = [impl](auto arg) { return impl->serve_process(arg); };
        _m_server = std::make_unique<WFHttpServer>(&server_params, proc);
        return StatusCode::OK;
    }

    std::unique_ptr<WFHttpServer> _m_server;
};
}
}

#endif //MORTRED_MODEL_SERVER_BASESERVER_H
