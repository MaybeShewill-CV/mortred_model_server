/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: supervisor.cpp
* Date: 26-8-22
************************************************/

// Thin executable wrapper: implementation lives in src/control.

#include "control/supervisor/supervisor_app.h"

int main() {
    return mortred::control::run_supervisor();
}
