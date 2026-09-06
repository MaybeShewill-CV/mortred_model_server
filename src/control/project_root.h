/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: project_root.h
* Date: 26-9-6
************************************************/

#ifndef MORTRED_CONTROL_PROJECT_ROOT_H
#define MORTRED_CONTROL_PROJECT_ROOT_H

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

namespace mortred {
namespace control {

/*** Locate the project root for the control-plane binaries (gateway /
 * supervisor). MORTRED_PROJECT_ROOT wins; otherwise walk up from
 * /proc/self/exe looking for an installed-tree marker (bin + deps), at most
 * 12 levels. Returns "." when nothing matches (tests / ad-hoc layouts).
 */
inline std::string resolve_project_root() {
    if (const char* env = std::getenv("MORTRED_PROJECT_ROOT"); env != nullptr && *env != '\0') {
        return env;
    }
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) {
        return ".";
    }
    buf[n] = '\0';
    std::filesystem::path p(buf);
    auto dir = p.parent_path();
    for (int i = 0; i < 12 && !dir.empty(); ++i) {
        std::error_code ec;
        const bool has_bin = std::filesystem::exists(dir / "_bin", ec) ||
                             std::filesystem::exists(dir / "bin", ec);
        ec.clear();
        const bool has_deps = std::filesystem::exists(dir / "3rd_party", ec) ||
                              std::filesystem::exists(dir / "lib", ec);
        if (has_bin && has_deps) {
            return dir.string();
        }
        dir = dir.parent_path();
    }
    return ".";
}

}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_PROJECT_ROOT_H
