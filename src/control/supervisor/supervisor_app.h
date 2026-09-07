/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: supervisor_app.h
* Date: 26-8-22
************************************************/

#ifndef MORTRED_CONTROL_SUPERVISOR_APP_H
#define MORTRED_CONTROL_SUPERVISOR_APP_H

#include <memory>
#include <string>

#include <workflow/WFHttpServer.h>

#include "control/catalog.h"
#include "control/control_config.h"
#include "control/supervisor.h"

namespace mortred {
namespace control {

/*** Explicit per-instance configuration. run() fills it from the process
 * environment (MORTRED_* / APP_* variables); tests construct it directly so
 * multiple SupervisorApp objects can coexist in one process. Empty string
 * fields mean "not configured"; empty config_path falls back to
 * <project_root>/conf/mortred.toml. api_token is mandatory (fail-closed). */
struct SupervisorInitOptions {
    std::string project_root;
    std::string config_path;
    std::string api_host;     // "" = config value
    int api_port = 0;         // 0 = config value
    std::string api_token;    // management bearer; never empty
    std::string metrics_token;         // scrape Bearer; required if gateway binary exists
    std::string gateway_auth_token;    // inference Bearer (MORTRED_GATEWAY_AUTH_TOKEN)
    std::string ui_dir;       // "" = <root>/share/mortred/ui | source ui
    std::string pack_path;    // "" = no machine pack
    int autostart_default = -1;  // -1 = keep config; 0/1 override
    std::string bin_dir;      // "" = config value
    std::string lib_dir;      // "" = config value
    std::string libs_dir;     // "" = config value
    std::string profile;      // "cpu"/"gpu"; "" = MORTRED_PROFILE / gpu
};

/*** Control-plane daemon: supervises mortred-gateway + the model servers,
 * serves the /api/v1 management surface and the embedded web UI. All state is
 * instance-local (the historical file-scope globals are gone); independent
 * instances may coexist in one process (tests rely on it). */
class SupervisorApp {
public:
    SupervisorApp() = default;
    ~SupervisorApp();

    SupervisorApp(const SupervisorApp&) = delete;
    SupervisorApp& operator=(const SupervisorApp&) = delete;

    // full daemon entry: blocks supervision signals, inits from env, exports
    // the child-inheritance env, listens, autostarts, serves until shutdown
    int run();

    // test surface: init runs every fail-closed startup check and brings the
    // ProcessSupervisor up (no autostart, no child-env export); listen /
    // stop_listen bracket a non-blocking serve window
    bool init(const SupervisorInitOptions& options);
    bool listen();
    void stop_listen();

private:
    void process(WFHttpTask* task);
    void handle_catalog(WFHttpTask* task);
    void handle_status(WFHttpTask* task);
    void handle_server_detail(WFHttpTask* task, const std::string& id);
    void handle_server_action(WFHttpTask* task, const std::string& id,
                              const std::string& action);
    void handle_logs(WFHttpTask* task, const std::string& id, const std::string& uri);
    void handle_metrics(WFHttpTask* task);
    void handle_graceful_restart(WFHttpTask* task, const std::string& server_id);
    void serve_static(WFHttpTask* task, const std::string& path);
    bool server_has_active_jobs(const std::string& server_id);

private:
    Catalog catalog_;
    ControlConfig cfg_;
    std::unique_ptr<ProcessSupervisor> supervisor_;
    std::string root_;
    std::string ui_dir_;
    std::string auth_token_;
    std::unique_ptr<WFHttpServer> server_;
};

/*** control-plane daemon entry (called by the thin main in src/apps) */
int run_supervisor();

}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_SUPERVISOR_APP_H
