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
    request_envelope_unittest
    blocking_worker_queue_unittest
    status_code_unittest
    auth_token_unittest
    time_stamp_unittest
    cv_utils_unittest
    detection_params_unittest
    detector_common_unittest
    model_runtime_unittest
    tensor_contract_unittest
    param_spec_unittest
    model_factory_unittest
    byte_tracker_unittest
    simple_tokenizer_unittest
    request_size_limit_unittest
    ready_probe_unittest
    rate_limiter_unittest
    cv_image_input_unittest
    worker_nums_unittest
    http_contract_test
    backpressure_unittest
    prometheus_metrics_test
    response_schema_test
    openapi_consistency_test
    config_schema_test
    async_job_unittest
    async_job_stress_test
    restart_policy_unittest
    catalog_unittest
    control_config_unittest
    supervisor_unittest
)

if(MORTRED_WORKFLOW_AVAILABLE)
    # HTTP-level e2e contract tests need to link workflow; if a tests-only build has no
    # vendored workflow lib, skip them (full builds / local dev run them).
    list(APPEND TEST_LIST server_e2e_contract_test)
endif()

# ApiKeyManager::authenticate needs OpenSSL SHA-256; the header is otherwise
# dependency-free, so the test is registered only when OpenSSL is linkable
# (preinstalled on the CI runners / dev boxes; skipped otherwise, like the
# workflow gate above).
find_package(OpenSSL QUIET)
if(OpenSSL_FOUND)
    list(APPEND TEST_LIST api_key_manager_unittest)
else()
    message(STATUS "OpenSSL not found: api_key_manager_unittest skipped")
endif()

if(MORTRED_BUILD_FULL)
    list(APPEND TEST_LIST gateway_e2e_test)
    list(APPEND TEST_LIST object_detection_output_contract_unittest)
    list(APPEND TEST_LIST model_output_contract_unittest)
    # scaffold canary: proves scripts/new_model.py templates still compile
    list(APPEND TEST_LIST rtdetr_detector_output_contract_unittest)
    # multi-session base class needs the real session factory (models lib)
    list(APPEND TEST_LIST multi_session_model_unittest)
    # SAM AMG param wiring pulls the encoder/decoder dtors from the models lib
    list(APPEND TEST_LIST sam_amg_param_unittest)
    list(APPEND TEST_LIST model_catalog_unittest)
    # GPU-model tests exercise TRT sessions and link vendored::mnn_cuda; the
    # cpu profile compiles TRT out, so they exist in the gpu profile only
    if(NOT MORTRED_BUILD_PROFILE STREQUAL "cpu")
        list(APPEND TEST_LIST model_golden_test)
        list(APPEND TEST_LIST model_lifecycle_unittest)
        list(APPEND TEST_LIST backend_unittest)
    endif()
endif()

# deterministic model-server stand-in used by supervision/gateway e2e tests
add_executable(fake_model_server EXCLUDE_FROM_ALL fake_model_server.cc)
target_link_libraries(fake_model_server PRIVATE Threads::Threads)

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
    elseif(${src} STREQUAL "tensor_contract_unittest")
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "detection_params_unittest")
        set(EXTRA_LIBS glog::glog)
    elseif(${src} STREQUAL "detector_common_unittest")
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "model_runtime_unittest")
        set(EXTRA_SRCS ${PROJECT_ROOT_DIR}/src/models/backend/model_runtime.cpp)
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "cv_image_input_unittest")
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "worker_nums_unittest")
        set(EXTRA_LIBS glog::glog)
    elseif(${src} STREQUAL "model_factory_unittest")
        # instantiates BaseAiServer (WFServer subclass): its vtable pulls
        # WFServerBase typeinfo from libworkflow, needed once sanitizer flags
        # make the compiler emit it; link the vendored lib when available
        set(EXTRA_LIBS glog::glog)
        if(MORTRED_WORKFLOW_AVAILABLE)
            list(APPEND EXTRA_LIBS vendored::workflow)
        endif()
    elseif(${src} STREQUAL "model_golden_test")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn vendored::mnn_cuda)
    elseif(${src} STREQUAL "model_lifecycle_unittest")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn vendored::mnn_cuda)
    elseif(${src} STREQUAL "object_detection_output_contract_unittest")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn)
    elseif(${src} STREQUAL "model_output_contract_unittest")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn)
    elseif(${src} STREQUAL "rtdetr_detector_output_contract_unittest")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn)
    elseif(${src} STREQUAL "multi_session_model_unittest")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn)
    elseif(${src} STREQUAL "sam_amg_param_unittest")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn)
    elseif(${src} STREQUAL "model_catalog_unittest")
        # factory links models + server transitively; opencv/mnn are needed by
        # the model headers pulled in through the task catalogs
        set(EXTRA_LIBS factory glog::glog ${OpenCV_LIBS} vendored::mnn)
    elseif(${src} STREQUAL "backend_unittest")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn vendored::mnn_cuda)
    elseif(${src} STREQUAL "simple_tokenizer_unittest")
        set(EXTRA_SRCS ${PROJECT_ROOT_DIR}/src/models/clip/simple_tokenizer.cpp)
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "ready_probe_unittest")
        set(EXTRA_LIBS pthread)
    elseif(${src} STREQUAL "async_job_unittest")
        set(EXTRA_LIBS pthread)
    elseif(${src} STREQUAL "api_key_manager_unittest")
        set(EXTRA_LIBS OpenSSL::Crypto pthread)
    elseif(${src} STREQUAL "async_job_stress_test")
        set(EXTRA_LIBS pthread)
    elseif(${src} STREQUAL "supervisor_unittest")
        set(EXTRA_LIBS control pthread)
    elseif(${src} STREQUAL "catalog_unittest")
        set(EXTRA_LIBS control pthread)
    elseif(${src} STREQUAL "control_config_unittest")
        set(EXTRA_LIBS control pthread)
    elseif(${src} STREQUAL "restart_policy_unittest")
        set(EXTRA_LIBS control pthread)
    elseif(${src} STREQUAL "gateway_e2e_test")
        set(EXTRA_LIBS control pthread)
    elseif(${src} STREQUAL "response_schema_test")
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "openapi_consistency_test")
        set(EXTRA_LIBS glog::glog)
    elseif(${src} STREQUAL "server_e2e_contract_test")
        set(EXTRA_LIBS glog::glog vendored::workflow pthread)
    endif()
    add_executable(${src} EXCLUDE_FROM_ALL ${src}.cc ${EXTRA_SRCS})
    target_link_libraries(${src} common ${EXTRA_LIBS} ${GTEST_LIBS})
    if(${src} STREQUAL "supervisor_unittest")
        target_compile_definitions(${src} PRIVATE
            MORTRED_FAKE_BIN_DEFAULT="$<TARGET_FILE:fake_model_server>")
    elseif(${src} STREQUAL "gateway_e2e_test")
        target_compile_definitions(${src} PRIVATE
            MORTRED_FAKE_BIN_DEFAULT="$<TARGET_FILE:fake_model_server>"
            MORTRED_GATEWAY_BIN_DEFAULT="$<TARGET_FILE:mortred-gateway.out>")
    endif()
    if(EXTRA_COMPILE_OPTIONS)
        target_compile_options(${src} PRIVATE ${EXTRA_COMPILE_OPTIONS})
    endif()
    add_test(NAME ${src} COMMAND $<TARGET_FILE:${src}> WORKING_DIRECTORY ${PROJECT_ROOT_DIR})
    add_dependencies(check ${src})
    if((${src} STREQUAL "async_job_unittest") OR (${src} STREQUAL "async_job_stress_test") OR
       (${src} STREQUAL "api_key_manager_unittest"))
        # the sanitizer CI gate runs exactly these under -fsanitize=thread
        set_tests_properties(${src} PROPERTIES LABELS "sanitizer")
    endif()
endforeach()

# e2e helper binaries: supervision tests need the fake model server; the
# gateway e2e test additionally needs the real gateway binary (full builds)
add_dependencies(supervisor_unittest fake_model_server)
if(MORTRED_BUILD_FULL)
    add_dependencies(gateway_e2e_test fake_model_server mortred-gateway.out)
endif()

if(CMAKE_MEMORYCHECK_COMMAND)
    foreach(src ${TEST_LIST})
        # Modern signature + generator expression to locate the binary (old ./${src} was relative to the project root and pointed at a nonexistent path)
        add_test(NAME ${src}-memory-check
                 COMMAND ${memcheck_command} $<TARGET_FILE:${src}>)
    endforeach()
endif()
