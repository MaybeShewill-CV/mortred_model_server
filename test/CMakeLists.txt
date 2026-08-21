cmake_minimum_required(VERSION 3.15)

project(mm_ai_server_test LANGUAGES C CXX)

find_program(CMAKE_MEMORYCHECK_COMMAND valgrind)
set(memcheck_command ${CMAKE_MEMORYCHECK_COMMAND} ${CMAKE_MEMORYCHECK_COMMAND_OPTIONS} --error-exitcode=1 --leak-check=full)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure)

find_package(GTest REQUIRED)

# GTest target names differ between module and config modes; pick a compatible set
if(TARGET GTest::GTest)
    set(GTEST_LIBS GTest::GTest GTest::Main)
elseif(TARGET GTest::gtest)
    set(GTEST_LIBS GTest::gtest GTest::gtest_main)
else()
    set(GTEST_LIBS ${GTEST_LIBRARIES})
endif()

if (WIN32)
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   /MP /wd4200")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /wd4200")
else ()
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -Wall -fPIC -pipe")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -pipe")
endif ()

set(TEST_LIST
    base64_unittest
    file_path_util_unittest
    json_request_parser_unittest
    blocking_worker_queue_unittest
    status_code_unittest
    time_stamp_unittest
    cv_utils_unittest
    model_factory_unittest
    byte_tracker_unittest
    simple_tokenizer_unittest
    request_size_limit_unittest
    web_console_auth_unittest
    ready_probe_unittest
    rate_limiter_unittest
    cv_image_input_unittest
    worker_nums_unittest
    http_contract_test
    prometheus_metrics_test
    response_schema_test
    openapi_consistency_test
    config_schema_test
)

if(MORTRED_WORKFLOW_AVAILABLE)
    # HTTP-level e2e contract tests need to link workflow; if a tests-only build has no
    # vendored workflow lib, skip them (full builds / local dev run them).
    list(APPEND TEST_LIST server_e2e_contract_test)
endif()

if(MORTRED_BUILD_FULL)
    list(APPEND TEST_LIST model_golden_test)
    list(APPEND TEST_LIST model_lifecycle_unittest)
    list(APPEND TEST_LIST backend_unittest)
endif()

foreach(src ${TEST_LIST})
    set(EXTRA_SRCS "")
    set(EXTRA_LIBS "")
    set(EXTRA_COMPILE_OPTIONS "")
    if(${src} STREQUAL "byte_tracker_unittest")
        set(EXTRA_SRCS
            ${PROJECT_ROOT_DIR}/src/models/mot/byte_tracker/byte_tracker.cpp
            ${PROJECT_ROOT_DIR}/src/models/mot/byte_tracker/kalman_filter.cpp
            ${PROJECT_ROOT_DIR}/src/models/mot/byte_tracker/lapjv.cpp
            ${PROJECT_ROOT_DIR}/src/models/mot/byte_tracker/strack.cpp)
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
        # lapjv.cpp is a vendored third-party solver; its pre-existing warnings skip the -Werror gate
        set(EXTRA_COMPILE_OPTIONS -Wno-unused-parameter)
    elseif(${src} STREQUAL "cv_utils_unittest")
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "cv_image_input_unittest")
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "worker_nums_unittest")
        set(EXTRA_LIBS glog::glog)
    elseif(${src} STREQUAL "model_factory_unittest")
        set(EXTRA_LIBS glog::glog)
    elseif(${src} STREQUAL "model_golden_test")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn vendored::mnn_cuda)
    elseif(${src} STREQUAL "model_lifecycle_unittest")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn vendored::mnn_cuda)
    elseif(${src} STREQUAL "backend_unittest")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn vendored::mnn_cuda)
    elseif(${src} STREQUAL "simple_tokenizer_unittest")
        set(EXTRA_SRCS ${PROJECT_ROOT_DIR}/src/models/clip/simple_tokenizer.cpp)
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "web_console_auth_unittest")
        set(EXTRA_SRCS ${PROJECT_ROOT_DIR}/src/apps/web_console/backend/auth.cpp)
    elseif(${src} STREQUAL "ready_probe_unittest")
        set(EXTRA_LIBS pthread)
    elseif(${src} STREQUAL "response_schema_test")
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "openapi_consistency_test")
        set(EXTRA_LIBS glog::glog)
    elseif(${src} STREQUAL "server_e2e_contract_test")
        set(EXTRA_LIBS glog::glog vendored::workflow pthread)
    endif()
    add_executable(${src} EXCLUDE_FROM_ALL ${src}.cc ${EXTRA_SRCS})
    target_link_libraries(${src} common ${EXTRA_LIBS} ${GTEST_LIBS})
    if(EXTRA_COMPILE_OPTIONS)
        target_compile_options(${src} PRIVATE ${EXTRA_COMPILE_OPTIONS})
    endif()
    add_test(NAME ${src} COMMAND $<TARGET_FILE:${src}> WORKING_DIRECTORY ${PROJECT_ROOT_DIR})
    add_dependencies(check ${src})
endforeach()

if(CMAKE_MEMORYCHECK_COMMAND)
    foreach(src ${TEST_LIST})
        # Modern signature + generator expression to locate the binary (old ./${src} was relative to the project root and pointed at a nonexistent path)
        add_test(NAME ${src}-memory-check
                 COMMAND ${memcheck_command} $<TARGET_FILE:${src}>)
    endforeach()
endif()
