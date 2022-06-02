/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: test.cpp
* Date: 22-6-2
************************************************/

#include <iostream>

#include <common/base64.h>
#include <common/status_code.h>

using morted::common::Base64;
using morted::common::StatusCode;

int main(int argc, char** argv) {
    std::cout << "hello world" << std::endl;
    std::cout << "status code: 0, means: " << morted::common::error_code_to_str(0) << std::endl;
    return 1;
}