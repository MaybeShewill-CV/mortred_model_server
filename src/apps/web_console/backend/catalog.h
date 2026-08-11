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
    std::string category; // classification / enhancement / ... / llm
    std::string exe;      // executable file name in _bin
    std::string config;   // absolute config path
    std::string host;
    int port = 0;
    std::string uri;      // server_uri or server_url
    std::string type;     // "image" or "chat"
    bool generated_config = false;
};

class Catalog {
  public:
    /***
     * scan _bin + conf/server to build the server registry
     * @param project_root project root dir
     * @param generated_dir dir for auto-generated configs
     * @return true on success
     */
    bool init(const std::string& project_root, const std::string& generated_dir);

    const std::vector<ServerEntry>& entries() const {
        return _entries;
    }

    const ServerEntry* find(const std::string& id) const;

  private:
    void add_missing_server(const std::string& project_root,
                            const std::string& generated_dir,
                            const std::string& exe,
                            const std::string& section,
                            const std::string& model_section,
                            int port,
                            const std::string& uri,
                            const std::string& category,
                            const std::string& model_cfg_rel);

    std::vector<ServerEntry> _entries;
};

} // namespace mortred_web

#endif // MORTRED_WEB_CATALOG_H
