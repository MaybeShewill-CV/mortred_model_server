cmake_minimum_required(VERSION 3.18)

project(mortred_model_server VERSION 1.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})

# version header consumed by mortredctl/supervisor --version; single source of
# truth = project(VERSION), which equals the git tag of the release pipeline.
# Generated into the build tree, never into src/: a committed generated header
# makes every checkout dirty after the first configure.
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/generated/common)
configure_file(${PROJECT_ROOT_DIR}/src/common/mortred_version.h.in
               ${CMAKE_BINARY_DIR}/generated/common/mortred_version.h @ONLY)

# Build type: default to Release only when unspecified; never override a user's Debug/RelWithDebInfo choice
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug/Release/RelWithDebInfo)" FORCE)
endif()

message(STATUS ">>>>>>>>>>>>>")
message(STATUS "Mortred Model Server INFO:")
message(STATUS "\tSystem: ${CMAKE_SYSTEM_NAME}")
message(STATUS "\tProcessor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "\tBuild Path: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "\tBuild Type: ${CMAKE_BUILD_TYPE}")

# Output dirs configurable; default _bin/_lib layout used by runtime binaries, mortredctl, and scripts
set(MORTRED_BIN_OUTPUT_DIR ${PROJECT_ROOT_DIR}/_bin CACHE PATH "Executable output directory")
set(MORTRED_LIB_OUTPUT_DIR ${PROJECT_ROOT_DIR}/_lib CACHE PATH "Shared library output directory")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MORTRED_BIN_OUTPUT_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${MORTRED_LIB_OUTPUT_DIR})

# CI / unit-test mode + profile must be known before vendored FATAL fix strings (SME-10).
option(MORTRED_BUILD_FULL "Build full project (models/server/apps, requires CUDA/MNN/TRT)" ON)
# Deployment profile: gpu = full CUDA/TensorRT stack (default, unchanged);
# cpu = MNN-CPU + ORT-CPU backends only, no TensorRT (the trt backend is
# compiled out; trt-configured models fail init with a clear error). Single
# source of truth for the deployment story - see docs/deployment.md.
set(MORTRED_BUILD_PROFILE "gpu" CACHE STRING "Deployment profile: gpu | cpu")
set_property(CACHE MORTRED_BUILD_PROFILE PROPERTY STRINGS gpu cpu)

# _bin/_lib default to one shared location at the repo root, so a second build
# directory configured with a different profile silently overwrites the first
# profile's binaries. Worse, the foreign binary is then newer than this
# profile's objects, so a later `cmake --build` of the right profile considers
# the target up-to-date and never relinks (observed twice: a cpu configure in
# build-eval replaced _bin/mortred-model-server.out, after which every
# TensorRT id failed to start with "tensorrt backend is not compiled into this
# build (cpu profile)", and the gpu rebuild printed no Linking line at all).
# Stamp the profile and refuse a conflicting configure in the same output dir.
set(_mortred_profile_stamp "${MORTRED_BIN_OUTPUT_DIR}/.mortred-build-profile")
file(MAKE_DIRECTORY "${MORTRED_BIN_OUTPUT_DIR}")
if(EXISTS "${_mortred_profile_stamp}")
    file(READ "${_mortred_profile_stamp}" _mortred_stamped_profile)
    string(STRIP "${_mortred_stamped_profile}" _mortred_stamped_profile)
    if(NOT _mortred_stamped_profile STREQUAL MORTRED_BUILD_PROFILE)
        message(FATAL_ERROR
            "output dir ${MORTRED_BIN_OUTPUT_DIR} already holds a ${_mortred_stamped_profile}-profile build; "
            "configuring ${MORTRED_BUILD_PROFILE} here would overwrite those binaries.\n"
            "Give the second profile its own output dirs, e.g.\n"
            "  cmake -S . -B build-cpu -DMORTRED_BUILD_PROFILE=cpu "
            "-DMORTRED_BIN_OUTPUT_DIR=<build-dir>/_bin -DMORTRED_LIB_OUTPUT_DIR=<build-dir>/_lib\n"
            "or delete ${_mortred_profile_stamp} if this tree is meant to switch profile.")
    endif()
else()
    file(WRITE "${_mortred_profile_stamp}" "${MORTRED_BUILD_PROFILE}\n")
endif()
option(MORTRED_INSTALL "Create the tree-style install target (bin/lib tree)" OFF)

# Vendored deps: exact soname + FATAL with profile-aware fix + auto-reconfigure
include(${PROJECT_ROOT_DIR}/cmake/VendoredDeps.cmake)
mortred_probe_workflow()
# control_workflow (gateway/supervisor app objects) is built whenever workflow
# is present, including tests-only on a machine that already ran
# install_deps.sh. ApiKeyManager SHA-256 needs the same libcrypto soname the
# install tree ships; importing it only inside MORTRED_BUILD_FULL left
# tests-only configure failing with "vendored::crypto was not found".
if(MORTRED_WORKFLOW_AVAILABLE)
    mortred_import_crypto()
endif()

# CI quality gate: treat compiler warnings as errors (enabled in tests-only builds
# and the Docker full-build gate). nvcc has its own -Werror=<kind> and treats the
# next token as that kind, so a global -Werror becomes:
#   nvcc fatal : Value '-std=c++17' is not defined for option 'Werror'
# on gpu_preprocess.cu. Keep the gate on C/CXX; do not pass it to CUDA.
option(MORTRED_ENABLE_WERROR "Treat compiler warnings as errors (used by CI quality gate)" OFF)
if(MORTRED_ENABLE_WERROR)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C>:-Wall>
        $<$<COMPILE_LANGUAGE:C>:-Wextra>
        $<$<COMPILE_LANGUAGE:C>:-Werror>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Werror>
    )
endif()

# find package
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)
find_package(Eigen3 REQUIRED)

# Common include scope: propagated PUBLICly by library targets instead of global include_directories.
# 3rd_party and system deps marked SYSTEM: vendored header warnings are exempt from the -Werror gate.
add_library(mortred_common_includes INTERFACE)
target_include_directories(mortred_common_includes INTERFACE
    ${PROJECT_ROOT_DIR}/src
    ${CMAKE_BINARY_DIR}/generated   # mortred_version.h (generated, not committed)
)
target_include_directories(mortred_common_includes SYSTEM INTERFACE
    ${PROJECT_ROOT_DIR}/3rd_party/include
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

# add subdirectory
add_subdirectory(src/common)
# control-plane module: supervision lib + gateway/supervisor/cli (see src/control)
add_subdirectory(src/control)

if(MORTRED_BUILD_FULL)
    # SME-10: fail closed at configure (clear fix cmds), not at obscure link/compile.
    mortred_require_workflow_for_full()
    # exact sonames pin versions (CUDA 12 / TensorRT 10.3 / ORT 1.29)
    mortred_import_shared(mnn            libMNN.so)
    mortred_import_shared(fmt            libfmt.so.9)
    mortred_import_shared(onnxruntime    libonnxruntime.so.1.29.0)
    mortred_assert_onnxruntime_pin()
    if(MORTRED_BUILD_PROFILE STREQUAL "cpu")
        message(STATUS "Build profile: cpu (MNN-CPU + ORT-CPU, TensorRT compiled out)")
    else()
        message(STATUS "Build profile: gpu (full CUDA 12 / TensorRT 10 stack)")
        string(TOUPPER "${CMAKE_BUILD_TYPE}" _mortred_bt)
        string(CONCAT _mortred_sanitize_flags
            "${CMAKE_C_FLAGS} ${CMAKE_CXX_FLAGS} "
            "${CMAKE_C_FLAGS_${_mortred_bt}} ${CMAKE_CXX_FLAGS_${_mortred_bt}} "
            "$ENV{CFLAGS} $ENV{CXXFLAGS}")
        if(_mortred_sanitize_flags MATCHES "-fsanitize=[^;\"]*address")
            message(FATAL_ERROR
                "AddressSanitizer is forbidden on the full GPU build "
                "(ASan + TensorRT/cudart mixed ELF). Rebuild Release without -fsanitize=address:\n"
                "  cmake --preset full && cmake --build --preset full")
        endif()
        file(GLOB _mortred_leftover_trt8 "${MORTRED_3RD_LIBS}/libnvinfer.so.8*")
        if(_mortred_leftover_trt8)
            message(FATAL_ERROR
                "leftover TensorRT 8 libraries in ${MORTRED_3RD_LIBS}:\n  ${_mortred_leftover_trt8}\n"
                "fix: sudo ./scripts/install_deps.sh --nvidia\n"
                "verify: ./scripts/install_deps.sh --check")
        endif()
        find_package(CUDAToolkit REQUIRED)
        if(CUDAToolkit_VERSION VERSION_LESS 12 OR CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13)
            message(FATAL_ERROR
                "CUDA Toolkit 12.x required, found ${CUDAToolkit_VERSION}")
        endif()
        mortred_import_shared(mnn_cuda       libMNN_Cuda_Main.so)
        mortred_import_shared(nvinfer        libnvinfer.so.10)
        mortred_import_shared(nvinfer_plugin libnvinfer_plugin.so.10)
        mortred_import_shared(nvonnxparser   libnvonnxparser.so.10)
    endif()
    add_subdirectory(src/models)
    add_subdirectory(src/factory)
    add_subdirectory(src/server)
    add_subdirectory(src/apps)

    # contract generation chain: C++ catalogs -> docs/contract_dump.json ->
    # gen_openapi.py -> openapi.json + openapi_doc.h (scripts/check_contract_sync.py)
    add_executable(contract_dump ${PROJECT_ROOT_DIR}/scripts/contract_dump.cc)
    target_include_directories(contract_dump PRIVATE ${PROJECT_ROOT_DIR}/src)
    target_include_directories(contract_dump SYSTEM PRIVATE ${PROJECT_ROOT_DIR}/3rd_party/include)
    target_link_libraries(contract_dump PRIVATE factory common)
endif()

# add unittest
enable_testing()
add_subdirectory(test EXCLUDE_FROM_ALL)

# Install target (enabled with MORTRED_INSTALL=ON): tree-style install of bin/lib.
# Runtime entry (Docker/systemd) sets LD_LIBRARY_PATH=/opt/mortred/lib explicitly;
# do not rely on the build's RUNPATH, which may contain absolute source/build-machine paths
# The Docker build stage depends on this target.
if(MORTRED_INSTALL)
    install(DIRECTORY ${MORTRED_BIN_OUTPUT_DIR}/ DESTINATION bin
            FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                             GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    install(DIRECTORY ${MORTRED_LIB_OUTPUT_DIR}/ DESTINATION lib)
    install(DIRECTORY ${PROJECT_ROOT_DIR}/3rd_party/libs/ DESTINATION lib)
    # trtexec (external engine-conversion CLI, copied by install_deps.sh --nvidia) lands in bin/
    if(EXISTS ${PROJECT_ROOT_DIR}/3rd_party/bin)
        install(DIRECTORY ${PROJECT_ROOT_DIR}/3rd_party/bin/ DESTINATION bin
                FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                 GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    endif()
    install(DIRECTORY conf docs demo_data resources DESTINATION .)
    install(DIRECTORY ${PROJECT_ROOT_DIR}/deploy/nginx DESTINATION deploy)
    install(DIRECTORY scripts DESTINATION .
            FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                             GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    install(DIRECTORY src/control/supervisor/ui DESTINATION share/mortred)
endif()
