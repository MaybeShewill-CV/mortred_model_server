/************************************************
 * Author: Codex
 * File: catalog.h
 ************************************************/

#ifndef MORTRED_WEB_CATALOG_H
#define MORTRED_WEB_CATALOG_H

#include <string>
#include <vector>

namespace mortred_web {

struct ServerEntry {
    std::string id;       // unique id (exe name without .out)
    std::string name;     // display name
    std::string category; // classification / enhancement / ...
    std::string exe;      // executable file name in _bin (declared by server_exe in conf)
    std::string config;   // absolute config path
    std::string host;
    std::string auth_token; // bearer token for direct inference access (empty = disabled)
    int port = 0;
    std::string uri;      // server_uri
    std::string type;     // "image" or "chat"
};

class Catalog {
  public:
    /***
     * Build the server registry from the TOML configs under conf/server only. Every
     * [*_SERVER] section must declare `server_exe` (plus port / host /
     * server_uri), so the config -> executable mapping is explicit and can
     * never silently go stale (enforced by scripts/check_consistency.py
     * check_server_exe_mapping).
     * @param project_root project root dir
     * @return true on success
     */
    bool init(const std::string& project_root);

    const std::vector<ServerEntry>& entries() const {
        return _entries;
    }

    const ServerEntry* find(const std::string& id) const;

  private:
    std::vector<ServerEntry> _entries;
};

}  // namespace mortred_web

#endif // MORTRED_WEB_CATALOG_H
