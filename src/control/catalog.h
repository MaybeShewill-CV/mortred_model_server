/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: catalog.h
* Date: 26-8-22
************************************************/

#ifndef MORTRED_CONTROL_CATALOG_H
#define MORTRED_CONTROL_CATALOG_H

#include <string>
#include <vector>

namespace mortred {
namespace control {

/***
 * One managed model server, derived from its conf/server TOML file.
 */
struct ServerEntry {
    std::string id;        // unique id: catalog model_section, or exe stem for test fakes
    std::string name;      // display name (== id)
    std::string category;  // conf/server sub-directory (classification / ...)
    std::string exe;       // executable file name in the bin dir
    std::string config;    // absolute config path
    std::string host;      // declared host (supervisor always binds children loopback)
    std::string model;     // catalog id; empty for legacy fake servers
    int port = 0;
    std::string uri;       // server_uri, the gateway routing key
    std::string type;      // "image" or "chat"
    std::string profile = "gpu";  // cpu | gpu | any (absent field = gpu: the
                                   // cpu catalog is explicitly curated)
};

inline constexpr const char* kUnifiedServerExe = "mortred-model-server.out";

/***
 * Server registry built from conf/server TOML configs. Single source of
 * truth shared by the supervisor and the gateway: every [*_SERVER] section
 * must declare port + server_uri. Product servers identify themselves with
 * `model = "YOLOV8"` (id = model, default exe mortred-model-server.out).
 * Test fakes omit `model` and keep an explicit server_exe; id is the exe stem.
 * Load fails on duplicate ids, duplicate ports or duplicate routing URIs.
 */
class Catalog {
  public:
    /***
     * @param project_root project root directory (contains conf/server)
     * @param err filled with the first fatal problem when returning false
     */
    /***
     * @param project_root project root directory (contains conf/server)
     * @param err filled with the first fatal problem when returning false
     * @param profile runtime deployment profile ("cpu" | "gpu", default gpu);
     *        entries whose profile field is not this value and not "any" are
     *        filtered out BEFORE the duplicate-id/port/uri checks, so cpu and
     *        gpu variants of the same model may reuse the same port
     */
    bool init(const std::string& project_root, std::string* err = nullptr,
              const std::string& profile = "gpu");

    const std::vector<ServerEntry>& entries() const {
        return _entries;
    }

    const ServerEntry* find(const std::string& id) const;

    const ServerEntry* find_by_uri(const std::string& uri) const;

  private:
    std::vector<ServerEntry> _entries;
};

}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_CATALOG_H
