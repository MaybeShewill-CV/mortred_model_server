/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: gateway_app.h
* Date: 26-8-22
************************************************/

#ifndef MORTRED_CONTROL_GATEWAY_APP_H
#define MORTRED_CONTROL_GATEWAY_APP_H

namespace mortred {
namespace control {

/*** data-plane reverse proxy entry (called by the thin main in src/apps) */
int run_gateway(int argc, char** argv);

}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_GATEWAY_APP_H
