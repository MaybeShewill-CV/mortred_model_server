/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: listen_port.h
* Date: 26-10-25
************************************************/

// The port is read from the model section or from MORTRED_LISTEN_PORT and later
// narrowed to unsigned short for the listen call, so a 0 or a 70000 has to be
// rejected here instead of silently becoming a different port.

#ifndef MORTRED_APPS_COMMON_LISTEN_PORT_H
#define MORTRED_APPS_COMMON_LISTEN_PORT_H

#include <cstdint>

namespace mortred {
namespace apps {

inline bool valid_listen_port(int64_t port) {
    return port > 0 && port <= 65535;
}

} // namespace apps
} // namespace mortred

#endif // MORTRED_APPS_COMMON_LISTEN_PORT_H
