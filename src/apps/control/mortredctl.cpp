/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: mortredctl.cpp
* Date: 26-8-22
************************************************/

// Thin executable wrapper: implementation lives in src/control.

#include "control/cli/cli_app.h"

int main(int argc, char** argv) {
    return mortred::control::run_cli(argc, argv);
}
