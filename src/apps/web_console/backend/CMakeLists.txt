cmake_minimum_required(VERSION 3.15)
project(mortred_web_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
include_directories("${PROJECT_ROOT}/3rd_party/include" "${PROJECT_ROOT}/src")

add_executable(app_server
    main.cpp
    catalog.cpp
    server_manager.cpp
    auth.cpp
)

target_compile_options(app_server PRIVATE -O2 -Wno-deprecated-declarations)
target_link_libraries(app_server workflow pthread)
target_link_directories(app_server PRIVATE "${PROJECT_ROOT}/3rd_party/libs")
target_link_options(app_server PRIVATE "-Wl,-rpath,${PROJECT_ROOT}/3rd_party/libs")
