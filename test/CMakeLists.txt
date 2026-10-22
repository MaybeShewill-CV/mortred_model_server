cmake_minimum_required(VERSION 3.15)

project(mm_ai_server_test LANGUAGES C CXX)

# Opt-in curated valgrind tests (A6). Default OFF so a machine with valgrind
# installed does not silently double the suite under check/ctest.
option(MORTRED_ENABLE_VALGRIND_TESTS
       "Register a small curated set of *-memory-check ctests (label=valgrind)"
       OFF)
find_program(CMAKE_MEMORYCHECK_COMMAND valgrind)
# Always use the in-tree suppression file (glog/libunwind .init Param noise).
set(_mortred_valgrind_supp "${CMAKE_CURRENT_SOURCE_DIR}/valgrind.supp")
set(memcheck_command
    ${CMAKE_MEMORYCHECK_COMMAND}
    ${CMAKE_MEMORYCHECK_COMMAND_OPTIONS}
    --error-exitcode=1
    --leak-check=full
    --suppressions=${_mortred_valgrind_supp})

# Default check excludes label=valgrind and label=heavy-golden.
# MORTRED_CHECK_HEAVY=1 includes heavy-golden and still excludes valgrind.
# Excluded ctest names, why, and alternate gates: test/CHECK_SSOT.md
# (fail-closed via scripts/check_check_ssot.py).
# JUnit lets scripts/ctest_summary.py see SKIP_RETURN_CODE 77 on tests that run.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21)
    set(_mortred_check_junit "${CMAKE_BINARY_DIR}/ctest-junit.xml")
else()
    set(_mortred_check_junit "")
endif()
add_custom_target(check
    COMMAND ${CMAKE_COMMAND}
        -DCTEST=${CMAKE_CTEST_COMMAND}
        -DTEST_DIR=${CMAKE_CURRENT_BINARY_DIR}
        -DJUNIT=${_mortred_check_junit}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_default_check.cmake
    COMMENT "default check excludes valgrind and heavy-golden unless MORTRED_CHECK_HEAVY=1"
    VERBATIM)

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
    mini_toml_unittest
    sync_request_graph_unittest
    gpu_sampler_unittest
    file_path_util_unittest
    request_envelope_unittest
    envelope_codec_unittest
    diffusion_param_unittest
    blocking_worker_queue_unittest
    status_code_unittest
    auth_token_unittest
    time_stamp_unittest
    cv_utils_unittest
    base_model_nvi_unittest
    packed_batch_nvi_unittest
    detection_params_unittest
    detector_common_unittest
    model_runtime_unittest
    tensor_contract_unittest
    session_io_unittest
    mnn_layout_unittest
    param_spec_unittest
    model_factory_unittest
    byte_tracker_unittest
    simple_tokenizer_unittest
    request_size_limit_unittest
    ready_probe_unittest
    cv_image_input_unittest
    worker_nums_unittest
    request_admission_unittest
    log_buffer_unittest
    server_runtime_config_unittest
    worker_pool_unittest
    batch_collector_unittest
    item_exec_unittest
    http_contract_test
    backpressure_unittest
    rate_limit_gcra_unittest
    rate_limit_subject_unittest
    prometheus_metrics_test
    response_schema_test
    openapi_consistency_test
    config_schema_test
    async_job_unittest
    async_job_stress_test
    restart_policy_unittest
    catalog_unittest
    control_config_unittest
    trt_spawn_gate_unittest
    occupancy_gate_unittest
    gpu_probe_unittest
    api_key_inflight_unittest
    trust_tokens_unittest
    supervisor_unittest
)

if(MORTRED_WORKFLOW_AVAILABLE)
    # HTTP-level e2e contract tests need to link workflow; if a tests-only build has no
    # vendored workflow lib, skip them (full builds / local dev run them).
    list(APPEND TEST_LIST server_e2e_contract_test)
    list(APPEND TEST_LIST process_stop_unittest)
    list(APPEND TEST_LIST do_work_lifetime_unittest)
    # in-process GatewayApp / SupervisorApp instances (control_workflow target
    # exists under the same condition)
    list(APPEND TEST_LIST gateway_multiinstance_test)
    list(APPEND TEST_LIST supervisor_multiinstance_test)
    list(APPEND TEST_LIST http_reply_unittest)
endif()

# ApiKeyManager::authenticate needs OpenSSL SHA-256; the header is otherwise
# dependency-free, so the test is registered only when OpenSSL is linkable
# (preinstalled on the CI runners / dev boxes; skipped otherwise, like the
# workflow gate above).
find_package(OpenSSL QUIET)
if(OpenSSL_FOUND)
    list(APPEND TEST_LIST api_key_manager_unittest)
    list(APPEND TEST_LIST rate_limit_authority_unittest)
    list(APPEND TEST_LIST tenant_quota_noisy_neighbor_unittest)
else()
    message(STATUS "OpenSSL not found: api_key_manager_unittest skipped")
endif()

if(MORTRED_BUILD_FULL)
    list(APPEND TEST_LIST gateway_e2e_test)
    list(APPEND TEST_LIST gateway_dual_upstream_isolation_test)
    list(APPEND TEST_LIST supervisor_malformed_e2e_test)
    list(APPEND TEST_LIST object_detection_output_contract_unittest)
    list(APPEND TEST_LIST model_output_contract_unittest)
    list(APPEND TEST_LIST dinov2_output_contract_unittest)
    list(APPEND TEST_LIST msocrnet_output_contract_unittest)
    list(APPEND TEST_LIST superpoint_output_contract_unittest)
    list(APPEND TEST_LIST enlightengan_output_contract_unittest)
    list(APPEND TEST_LIST realesrgan_output_contract_unittest)
    list(APPEND TEST_LIST attentive_gan_output_contract_unittest)
    list(APPEND TEST_LIST lightglue_output_contract_unittest)
    list(APPEND TEST_LIST sam_output_contract_unittest)
    list(APPEND TEST_LIST openai_clip_output_contract_unittest)
    list(APPEND TEST_LIST hrnet_output_contract_unittest)
    list(APPEND TEST_LIST mono_depth_output_contract_unittest)
    # scaffold canary: proves scripts/new_model.py templates still compile
    list(APPEND TEST_LIST rtdetr_detector_output_contract_unittest)
    # multi-session base class needs the real session factory (models lib)
    list(APPEND TEST_LIST multi_session_model_unittest)
    # SAM AMG param wiring pulls the encoder/decoder dtors from the models lib
    list(APPEND TEST_LIST sam_amg_param_unittest)
    list(APPEND TEST_LIST model_catalog_unittest)
    # Session / lifecycle / golden need the models lib. They used to be
    # gpu-profile-only because the test binaries linked vendored::mnn_cuda
    # directly; models already PUBLIC-links that on gpu and omits it on cpu.
    # TRT-only TEST()s are compiled out via MORTRED_HAS_TRT.
    list(APPEND TEST_LIST model_golden_test)
    list(APPEND TEST_LIST model_lifecycle_unittest)
    list(APPEND TEST_LIST backend_unittest)
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
    elseif(${src} STREQUAL "base_model_nvi_unittest")
        set(EXTRA_LIBS glog::glog)
    elseif(${src} STREQUAL "packed_batch_nvi_unittest")
        # BackendCvModel embeds GpuDecodeSlot; do not link libmodels (own create).
        set(EXTRA_SRCS ${PROJECT_ROOT_DIR}/src/models/backend/gpu_jpeg_decoder.cpp)
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "tensor_contract_unittest")
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "session_io_unittest")
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "mnn_layout_unittest")
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "detection_params_unittest")
        set(EXTRA_LIBS glog::glog)
    elseif(${src} STREQUAL "detector_common_unittest")
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "model_runtime_unittest")
        set(EXTRA_SRCS ${PROJECT_ROOT_DIR}/src/models/backend/model_runtime.cpp)
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "cv_image_input_unittest")
        set(EXTRA_SRCS ${PROJECT_ROOT_DIR}/src/models/backend/gpu_jpeg_decoder.cpp)
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "worker_nums_unittest")
        set(EXTRA_LIBS glog::glog)
    elseif(${src} STREQUAL "server_runtime_config_unittest")
        set(EXTRA_LIBS glog::glog)
    elseif(${src} STREQUAL "worker_pool_unittest")
        set(EXTRA_LIBS glog::glog)
    elseif(${src} STREQUAL "rate_limit_gcra_unittest")
        # pure GCRA kernel: no glog/opencv, only std threads
        set(EXTRA_LIBS pthread)
    elseif(${src} STREQUAL "rate_limit_subject_unittest")
        # pure subject derivation: no glog/opencv/pthreads
        set(EXTRA_LIBS )
    elseif(${src} STREQUAL "batch_collector_unittest")
        set(EXTRA_LIBS glog::glog)
    elseif(${src} STREQUAL "item_exec_unittest")
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "model_factory_unittest")
        # instantiates BaseAiServer (WFServer subclass): its vtable pulls
        # WFServerBase typeinfo from libworkflow, needed once sanitizer flags
        # make the compiler emit it; link the vendored lib when available
        set(EXTRA_LIBS glog::glog)
        if(MORTRED_WORKFLOW_AVAILABLE)
            list(APPEND EXTRA_LIBS vendored::workflow)
        endif()
    elseif(${src} STREQUAL "diffusion_param_unittest")
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "model_golden_test")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "model_lifecycle_unittest")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "object_detection_output_contract_unittest")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn)
    elseif(${src} STREQUAL "model_output_contract_unittest")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn)
    elseif(${src} STREQUAL "dinov2_output_contract_unittest")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn)
    elseif(${src} STREQUAL "msocrnet_output_contract_unittest")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn)
    elseif(${src} STREQUAL "superpoint_output_contract_unittest")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn)
    elseif(${src} STREQUAL "enlightengan_output_contract_unittest")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn)
    elseif(${src} STREQUAL "realesrgan_output_contract_unittest")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn)
    elseif(${src} STREQUAL "attentive_gan_output_contract_unittest")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn)
    elseif(${src} STREQUAL "lightglue_output_contract_unittest")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn)
    elseif(${src} STREQUAL "sam_output_contract_unittest")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn)
    elseif(${src} STREQUAL "openai_clip_output_contract_unittest")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn)
    elseif(${src} STREQUAL "hrnet_output_contract_unittest")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} vendored::mnn)
    elseif(${src} STREQUAL "mono_depth_output_contract_unittest")
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
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "simple_tokenizer_unittest")
        set(EXTRA_SRCS ${PROJECT_ROOT_DIR}/src/models/clip/simple_tokenizer.cpp)
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "ready_probe_unittest")
        set(EXTRA_LIBS pthread)
    elseif(${src} STREQUAL "async_job_unittest")
        set(EXTRA_LIBS glog::glog pthread)
        if(MORTRED_WORKFLOW_AVAILABLE)
            list(APPEND EXTRA_LIBS vendored::workflow)
        endif()
    elseif(${src} STREQUAL "api_key_manager_unittest")
        set(EXTRA_LIBS OpenSSL::Crypto pthread)
    elseif(${src} STREQUAL "rate_limit_authority_unittest")
        set(EXTRA_LIBS OpenSSL::Crypto pthread)
    elseif(${src} STREQUAL "tenant_quota_noisy_neighbor_unittest")
        set(EXTRA_LIBS OpenSSL::Crypto pthread)
    elseif(${src} STREQUAL "async_job_stress_test")
        set(EXTRA_LIBS pthread)
    elseif(${src} STREQUAL "supervisor_unittest")
        set(EXTRA_LIBS control pthread)
    elseif(${src} STREQUAL "catalog_unittest")
        set(EXTRA_LIBS control pthread)
    elseif(${src} STREQUAL "control_config_unittest")
        set(EXTRA_LIBS control pthread)
    elseif(${src} STREQUAL "trt_spawn_gate_unittest")
        set(EXTRA_LIBS control pthread)
    elseif(${src} STREQUAL "occupancy_gate_unittest")
        set(EXTRA_LIBS control pthread)
    elseif(${src} STREQUAL "gpu_probe_unittest")
        set(EXTRA_LIBS control pthread)
    elseif(${src} STREQUAL "trust_tokens_unittest")
        set(EXTRA_LIBS control pthread)
    elseif(${src} STREQUAL "restart_policy_unittest")
        set(EXTRA_LIBS control pthread)
    elseif(${src} STREQUAL "gateway_e2e_test")
        # the auth fixture hashes keys via ApiKeyManager::sha256_hex (OpenSSL
        # SHA-256 inlined from the header), same link need as api_key_manager_unittest
        set(EXTRA_LIBS control OpenSSL::Crypto pthread)
    elseif(${src} STREQUAL "gateway_dual_upstream_isolation_test")
        set(EXTRA_LIBS control OpenSSL::Crypto pthread)
    elseif(${src} STREQUAL "supervisor_malformed_e2e_test")
        set(EXTRA_LIBS pthread)
    elseif(${src} STREQUAL "gateway_multiinstance_test")
        set(EXTRA_LIBS control_workflow pthread)
    elseif(${src} STREQUAL "supervisor_multiinstance_test")
        set(EXTRA_LIBS control_workflow pthread)
    elseif(${src} STREQUAL "response_schema_test")
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "gpu_sampler_unittest")
        set(EXTRA_LIBS control pthread)
    elseif(${src} STREQUAL "openapi_consistency_test")
        set(EXTRA_LIBS glog::glog)
    elseif(${src} STREQUAL "server_e2e_contract_test")
        set(EXTRA_LIBS glog::glog vendored::workflow pthread)
    elseif(${src} STREQUAL "process_stop_unittest")
        set(EXTRA_LIBS vendored::workflow pthread)
    elseif(${src} STREQUAL "http_reply_unittest")
        set(EXTRA_LIBS vendored::workflow pthread)
    elseif(${src} STREQUAL "do_work_lifetime_unittest")
        # backend_cv_model.h inlines a call to gpu_jpeg::render_decode_metrics();
        # compile that TU here instead of linking libmodels (own create).
        set(EXTRA_SRCS ${PROJECT_ROOT_DIR}/src/models/backend/gpu_jpeg_decoder.cpp)
        set(EXTRA_LIBS glog::glog vendored::workflow pthread ${OpenCV_LIBS})
    endif()
    set(THIS_GTEST_LIBS ${GTEST_LIBS})
    if(${src} STREQUAL "model_golden_test")
        # Own main sets the process exit after RUN_ALL_TESTS. Do not link gtest_main.
        if(TARGET GTest::gtest)
            set(THIS_GTEST_LIBS GTest::gtest)
        elseif(TARGET GTest::GTest)
            set(THIS_GTEST_LIBS GTest::GTest)
        endif()
    endif()
    add_executable(${src} EXCLUDE_FROM_ALL ${src}.cc ${EXTRA_SRCS})
    target_link_libraries(${src} common ${EXTRA_LIBS} ${THIS_GTEST_LIBS})
    if(${src} STREQUAL "supervisor_unittest")
        target_compile_definitions(${src} PRIVATE
            MORTRED_FAKE_BIN_DEFAULT="$<TARGET_FILE:fake_model_server>")
    elseif(${src} STREQUAL "supervisor_multiinstance_test")
        target_compile_definitions(${src} PRIVATE
            MORTRED_FAKE_BIN_DEFAULT="$<TARGET_FILE:fake_model_server>")
        add_dependencies(${src} fake_model_server)
    elseif(${src} STREQUAL "gateway_e2e_test")
        target_compile_definitions(${src} PRIVATE
            MORTRED_FAKE_BIN_DEFAULT="$<TARGET_FILE:fake_model_server>"
            MORTRED_GATEWAY_BIN_DEFAULT="$<TARGET_FILE:mortred-gateway.out>")
    elseif(${src} STREQUAL "gateway_dual_upstream_isolation_test")
        target_compile_definitions(${src} PRIVATE
            MORTRED_FAKE_BIN_DEFAULT="$<TARGET_FILE:fake_model_server>"
            MORTRED_GATEWAY_BIN_DEFAULT="$<TARGET_FILE:mortred-gateway.out>")
    elseif(${src} STREQUAL "supervisor_malformed_e2e_test")
        target_compile_definitions(${src} PRIVATE
            MORTRED_SUPERVISOR_BIN_DEFAULT="$<TARGET_FILE:mortred-supervisor.out>")
    endif()
    if(EXTRA_COMPILE_OPTIONS)
        target_compile_options(${src} PRIVATE ${EXTRA_COMPILE_OPTIONS})
    endif()
    add_test(NAME ${src} COMMAND $<TARGET_FILE:${src}> WORKING_DIRECTORY ${PROJECT_ROOT_DIR})
    add_dependencies(check ${src})
    if((${src} STREQUAL "async_job_unittest") OR (${src} STREQUAL "async_job_stress_test") OR
       (${src} STREQUAL "api_key_manager_unittest") OR (${src} STREQUAL "do_work_lifetime_unittest") OR
       (${src} STREQUAL "worker_pool_unittest") OR (${src} STREQUAL "batch_collector_unittest") OR
       (${src} STREQUAL "rate_limit_gcra_unittest"))
        # the sanitizer CI gate runs exactly these under -fsanitize=thread
        # SSOT: CI TSan --target list must match these labels (enforced by scripts/check_tsan_targets.py)
        set_tests_properties(${src} PROPERTIES LABELS "sanitizer")
    endif()
endforeach()

# ctest -R matches the CTest name, not the gtest suite. Alias the FULL
# binary so `-R RealEsrGanOutputContract` selects this contract suite.
if(TARGET realesrgan_output_contract_unittest)
    add_test(
        NAME RealEsrGanOutputContract
        COMMAND $<TARGET_FILE:realesrgan_output_contract_unittest> --gtest_filter=RealEsrGanOutputContract.*
        WORKING_DIRECTORY ${PROJECT_ROOT_DIR})
    add_dependencies(check realesrgan_output_contract_unittest)
endif()
if(TARGET attentive_gan_output_contract_unittest)
    add_test(
        NAME AttentiveGanOutputContract
        COMMAND $<TARGET_FILE:attentive_gan_output_contract_unittest> --gtest_filter=AttentiveGanOutputContract.*
        WORKING_DIRECTORY ${PROJECT_ROOT_DIR})
    add_dependencies(check attentive_gan_output_contract_unittest)
endif()
if(TARGET lightglue_output_contract_unittest)
    add_test(
        NAME LightGlueOutputContract
        COMMAND $<TARGET_FILE:lightglue_output_contract_unittest> --gtest_filter=LightGlueOutputContract.*
        WORKING_DIRECTORY ${PROJECT_ROOT_DIR})
    add_dependencies(check lightglue_output_contract_unittest)
endif()
if(TARGET sam_output_contract_unittest)
    add_test(
        NAME SamOutputContract
        COMMAND $<TARGET_FILE:sam_output_contract_unittest> --gtest_filter=SamOutputContract.*
        WORKING_DIRECTORY ${PROJECT_ROOT_DIR})
    add_dependencies(check sam_output_contract_unittest)
endif()
if(TARGET openai_clip_output_contract_unittest)
    add_test(
        NAME OpenAiClipOutputContract
        COMMAND $<TARGET_FILE:openai_clip_output_contract_unittest> --gtest_filter=OpenAiClipOutputContract.*
        WORKING_DIRECTORY ${PROJECT_ROOT_DIR})
    add_dependencies(check openai_clip_output_contract_unittest)
endif()
if(TARGET multi_session_model_unittest)
    add_test(
        NAME MultiSessionModel
        COMMAND $<TARGET_FILE:multi_session_model_unittest> --gtest_filter=MultiSessionModel.*
        WORKING_DIRECTORY ${PROJECT_ROOT_DIR})
    add_dependencies(check multi_session_model_unittest)
endif()
if(TARGET diffusion_param_unittest)
    add_test(
        NAME DiffusionParam
        COMMAND $<TARGET_FILE:diffusion_param_unittest> --gtest_filter=DiffusionParam.*
        WORKING_DIRECTORY ${PROJECT_ROOT_DIR})
    add_dependencies(check diffusion_param_unittest)
endif()
if(TARGET hrnet_output_contract_unittest)
    add_test(
        NAME HRNetOutputContract
        COMMAND $<TARGET_FILE:hrnet_output_contract_unittest> --gtest_filter=HRNetOutputContract.*
        WORKING_DIRECTORY ${PROJECT_ROOT_DIR})
    add_dependencies(check hrnet_output_contract_unittest)
endif()
if(TARGET object_detection_output_contract_unittest)
    add_test(
        NAME YoloV7OutputContract
        COMMAND $<TARGET_FILE:object_detection_output_contract_unittest> --gtest_filter=YoloV7OutputContract.*
        WORKING_DIRECTORY ${PROJECT_ROOT_DIR})
    add_dependencies(check object_detection_output_contract_unittest)
endif()
if(TARGET model_golden_test)
    add_test(
        NAME Yolov8TrtDetection
        COMMAND $<TARGET_FILE:model_golden_test> --gtest_filter=Yolov8TrtDetection.*
        WORKING_DIRECTORY ${PROJECT_ROOT_DIR})
    # ctest -R matches the CTest name. The CPU case lives inside model_golden_test,
    # so list it under a name that contains yolov8_detection.
    add_test(
        NAME yolov8_detection
        COMMAND $<TARGET_FILE:model_golden_test> --gtest_filter=model_golden.yolov8_detection
        WORKING_DIRECTORY ${PROJECT_ROOT_DIR})
    add_dependencies(check model_golden_test)
endif()

# SSOT: test/CHECK_SSOT.md rows with label heavy-golden. scripts/check_check_ssot.py
# fails if this list, the label, or RUN_SERIAL drifts. Applied only on a full
# build, which is when model_golden_test and its aliases are registered.
# tests-only does not register them; default check still passes -LE heavy-golden.
set(MORTRED_HEAVY_GOLDEN_TESTS
    model_golden_test
    Yolov8TrtDetection
    yolov8_detection
)
if(MORTRED_BUILD_FULL)
    foreach(heavy ${MORTRED_HEAVY_GOLDEN_TESTS})
        if(NOT TEST ${heavy})
            message(FATAL_ERROR
                "heavy-golden ctest '${heavy}' is listed in MORTRED_HEAVY_GOLDEN_TESTS "
                "but was not registered. See test/CHECK_SSOT.md.")
        endif()
        # SKIP_RETURN_CODE 77: missing weights are a skip when someone opts in
        # with MORTRED_CHECK_HEAVY=1. Hosted CI does not use that path; it runs
        # the binary with MORTRED_CI_REQUIRE_WEIGHTS=1 and ci_assert_gtest_xml.py.
        set_tests_properties(${heavy} PROPERTIES
            SKIP_RETURN_CODE 77
            RUN_SERIAL TRUE
            LABELS "heavy-golden")
    endforeach()
endif()

# P0-1 fail-closed black box. Script owns PASS/FAIL/SKIP and exits 0 when
# FAILS=0 (AC8 skip does not fail the process). No valgrind or heavy-golden
# label, so the default check target runs it. RUN_SERIAL: the script
# rewrites stub scripts under scripts/ and restores them on exit.
add_test(
    NAME p0_1_engine_fail_closed
    COMMAND bash ${PROJECT_ROOT_DIR}/scripts/test_p0_1_engine_fail_closed.sh
    WORKING_DIRECTORY ${PROJECT_ROOT_DIR})
set_tests_properties(p0_1_engine_fail_closed PROPERTIES RUN_SERIAL TRUE)

# P2-4 resident pair rules. --check only: no model processes and no live json.
# No valgrind or heavy-golden label, so the default check target runs it.
# The live run is not registered.
add_test(
    NAME resident_pair_record_rules
    COMMAND python3 scripts/resident_pair_record.py --check
    WORKING_DIRECTORY ${PROJECT_ROOT_DIR})

# P2-6 stop order rules. --check only: no supervisor, gateway, or model
# processes and no live json. No valgrind or heavy-golden label, so the
# default check target runs it. The live run is not registered.
add_test(
    NAME stop_order_record_rules
    COMMAND python3 scripts/stop_order_record.py --check
    WORKING_DIRECTORY ${PROJECT_ROOT_DIR})

# Prometheus scrape credential consistency. A rotated MORTRED_METRICS_TOKEN
# that never reached conf/local/mortred_metrics_token (or the bind-mounted
# copy) makes every scrape 401 with no other signal, so verify the pairing.
# --self-test needs no live token or file: it exercises matching, stale,
# missing and unset cases in a temp dir.
add_test(
    NAME prometheus_credentials_check
    COMMAND bash scripts/write_prometheus_credentials.sh --self-test
    WORKING_DIRECTORY ${PROJECT_ROOT_DIR})

# ctest skip visibility: SKIP_RETURN_CODE 77 must never look like a pass.
add_test(
    NAME ctest_summary_self_test
    COMMAND python3 scripts/ctest_summary.py --self-test
    WORKING_DIRECTORY ${PROJECT_ROOT_DIR})

# API key hygiene: unsalted SHA-256 at rest means the SECRET carries the
# entropy, so minted keys must be random and the documented weak example hash
# must never reach a config file.
add_test(
    NAME api_keys_hashing_check
    COMMAND bash scripts/check_api_keys_hashing.sh --self-test
    WORKING_DIRECTORY ${PROJECT_ROOT_DIR})
add_test(
    NAME mint_api_key_self_test
    COMMAND bash scripts/mint_api_key.sh --self-test
    WORKING_DIRECTORY ${PROJECT_ROOT_DIR})

# Manual GCRA performance baseline (PR-1 acceptance reference; NOT a CI test)
add_executable(rate_limit_gcra_bench EXCLUDE_FROM_ALL rate_limit_gcra_bench.cc)
# common carries the mortred_common_includes INTERFACE (src/ include root)
target_link_libraries(rate_limit_gcra_bench common pthread)

# e2e helper binaries: supervision tests need the fake model server; the
# gateway e2e test additionally needs the real gateway binary (full builds)
add_dependencies(supervisor_unittest fake_model_server)
if(MORTRED_BUILD_FULL)
    add_dependencies(gateway_e2e_test fake_model_server mortred-gateway.out)
    add_dependencies(gateway_dual_upstream_isolation_test fake_model_server mortred-gateway.out)
    add_dependencies(supervisor_malformed_e2e_test mortred-supervisor.out)
endif()

# Curated, fast, weight-free targets only — never the full TEST_LIST.
set(MORTRED_VALGRIND_TEST_TARGETS
    status_code_unittest
    base64_unittest
    param_spec_unittest
    tensor_contract_unittest
    session_io_unittest
)

if(MORTRED_ENABLE_VALGRIND_TESTS)
    if(NOT CMAKE_MEMORYCHECK_COMMAND)
        message(FATAL_ERROR
            "MORTRED_ENABLE_VALGRIND_TESTS=ON but valgrind was not found "
            "(CMAKE_MEMORYCHECK_COMMAND empty)")
    endif()
    foreach(src ${MORTRED_VALGRIND_TEST_TARGETS})
        if(NOT TARGET ${src})
            message(FATAL_ERROR "valgrind target missing from this build: ${src}")
        endif()
        # Modern signature + generator expression to locate the binary.
        add_test(NAME ${src}-memory-check
                 COMMAND ${memcheck_command} $<TARGET_FILE:${src}>)
        set_tests_properties(${src}-memory-check PROPERTIES LABELS "valgrind")
    endforeach()
    message(STATUS "Valgrind memory-check tests enabled (label=valgrind; "
                   "run: ctest -L valgrind --output-on-failure)")
endif()
