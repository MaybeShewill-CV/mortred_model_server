cmake_minimum_required(VERSION 3.15)
project(mortred_web_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)

add_executable(app_server
    main.cpp
    catalog.cpp
    server_manager.cpp
)

target_include_directories(app_server SYSTEM PRIVATE "${PROJECT_ROOT}/3rd_party/include")
target_include_directories(app_server PRIVATE "${PROJECT_ROOT}/src")
# 不再硬编码 -O2：让 CMAKE_BUILD_TYPE 决定优化级别；链接 imported workflow target
target_compile_options(app_server PRIVATE -Wno-deprecated-declarations)
target_link_libraries(app_server PRIVATE vendored::workflow Threads::Threads)
# 运行期在 3rd_party/libs 找 libworkflow/libssl（构建期 RPATH，非安装期）
set_target_properties(app_server PROPERTIES
    BUILD_RPATH "${PROJECT_ROOT}/3rd_party/libs")
