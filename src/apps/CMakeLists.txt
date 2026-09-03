# build apps

function(add_mortred_unified_app target)
    add_executable(${target}
        ${CMAKE_CURRENT_LIST_DIR}/product_index.cpp
        ${CMAKE_CURRENT_LIST_DIR}/benchmark/custom_drivers.cpp
        ${ARGN})
    target_link_libraries(${target} common models factory vendored::workflow glog::glog)
    target_compile_options(${target} PRIVATE -Wno-deprecated-declarations)
    set_target_properties(${target} PROPERTIES
        BUILD_RPATH "${PROJECT_ROOT_DIR}/3rd_party/libs;${PROJECT_ROOT_DIR}/_lib")
endfunction()

# Unified HTTP + benchmark entrypoints. ProductIndex is a projection of the
# factory catalogs, so a new catalog row does not need a new ELF or CMake line.
add_mortred_unified_app(mortred-model-server.out
    ${CMAKE_CURRENT_LIST_DIR}/model_server_main.cpp)
add_mortred_unified_app(mortred-model-benchmark.out
    ${CMAKE_CURRENT_LIST_DIR}/model_benchmark_main.cpp)

# --- control-plane executables ---
# src/apps produces every executable; the implementations live in src/control
# (thin mains here + *_app.cpp compiled in, mirroring the models/.inl pattern).
if(MORTRED_WORKFLOW_AVAILABLE)
    add_executable(mortred-gateway.out
        ${CMAKE_CURRENT_LIST_DIR}/control/gateway.cpp
        ${PROJECT_ROOT_DIR}/src/control/gateway/gateway_app.cpp)
    target_include_directories(mortred-gateway.out PRIVATE ${PROJECT_ROOT_DIR}/src)
    target_include_directories(mortred-gateway.out SYSTEM PRIVATE ${PROJECT_ROOT_DIR}/3rd_party/include)
    target_link_libraries(mortred-gateway.out PRIVATE control vendored::workflow vendored::crypto Threads::Threads)

    add_executable(mortred-supervisor.out
        ${CMAKE_CURRENT_LIST_DIR}/control/supervisor.cpp
        ${PROJECT_ROOT_DIR}/src/control/supervisor/supervisor_app.cpp)
    target_include_directories(mortred-supervisor.out PRIVATE ${PROJECT_ROOT_DIR}/src)
    target_include_directories(mortred-supervisor.out SYSTEM PRIVATE ${PROJECT_ROOT_DIR}/3rd_party/include)
    target_link_libraries(mortred-supervisor.out PRIVATE control common vendored::workflow Threads::Threads)

    add_executable(mortredctl.out
        ${CMAKE_CURRENT_LIST_DIR}/control/mortredctl.cpp
        ${PROJECT_ROOT_DIR}/src/control/cli/cli_app.cpp)
    target_include_directories(mortredctl.out PRIVATE ${PROJECT_ROOT_DIR}/src)
    target_include_directories(mortredctl.out SYSTEM PRIVATE ${PROJECT_ROOT_DIR}/3rd_party/include)
    target_link_libraries(mortredctl.out PRIVATE control common vendored::workflow Threads::Threads)

    foreach(control_bin mortred-gateway.out mortred-supervisor.out mortredctl.out)
        set_target_properties(${control_bin} PROPERTIES
            BUILD_RPATH "${PROJECT_ROOT_DIR}/3rd_party/libs;${PROJECT_ROOT_DIR}/_lib")
    endforeach()
endif()
