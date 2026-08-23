/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: supervisor.cpp
* Date: 26-8-22
************************************************/

// Thin executable wrapper: implementation lives in src/control.

#include <cstdio>
#include <cstring>

#include "common/mortred_version.h"
#include "control/supervisor/supervisor_app.h"

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--version") == 0) {
            std::fprintf(stdout, "mortred_model_server %s\n", MORTRED_VERSION);
            return 0;
        }
    }
    return mortred::control::run_supervisor();
}
