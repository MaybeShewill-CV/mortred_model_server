/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: cli_app.h
* Date: 26-8-22
************************************************/

#ifndef MORTRED_CONTROL_CLI_APP_H
#define MORTRED_CONTROL_CLI_APP_H

namespace mortred {
namespace control {

/*** mortredctl CLI entry (called by the thin main in src/apps) */
int run_cli(int argc, char** argv);

}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_CLI_APP_H
