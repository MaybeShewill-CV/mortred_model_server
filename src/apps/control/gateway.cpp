/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: gateway.cpp
* Date: 26-8-22
************************************************/

// Thin executable wrapper: implementation lives in src/control.

#include "control/gateway/gateway_app.h"

int main(int argc, char** argv) {
    return mortred::control::run_gateway(argc, argv);
}
