/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: exit_codes.h
* Date: 26-10-25
************************************************/

// Process exit codes for the app entry points. The values are the BSD sysexits
// ones, so a wrapper script can tell "the operator typed the command wrong" from
// "the config is wrong" from "the backend failed" without parsing stderr.

#ifndef MORTRED_APPS_COMMON_EXIT_CODES_H
#define MORTRED_APPS_COMMON_EXIT_CODES_H

namespace mortred {
namespace apps {

constexpr int kExitUsage = 64;    // EX_USAGE: bad argv, unknown --model, missing args
constexpr int kExitSoftware = 70; // EX_SOFTWARE: backend/session init or listen failed
constexpr int kExitConfig = 78;   // EX_CONFIG: unparsable or inconsistent config

} // namespace apps
} // namespace mortred

#endif // MORTRED_APPS_COMMON_EXIT_CODES_H
