cmake_minimum_required(VERSION 3.18)

project(mortred_model_server VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})

# version header consumed by mortredctl/supervisor --version; single source of
# truth = project(VERSION), which equals the git tag of the release pipeline
configure_file(${PROJECT_ROOT_DIR}/src/common/mortred_version.h.in
               ${PROJECT_ROOT_DIR}/src/common/mortred_version.h @ONLY)

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

# Output dirs configurable; default to legacy _bin/_lib layout (web_console/ServerManager depends on it)
set(MORTRED_BIN_OUTPUT_DIR ${PROJECT_ROOT_DIR}/_bin CACHE PATH "Executable output directory")
set(MORTRED_LIB_OUTPUT_DIR ${PROJECT_ROOT_DIR}/_lib CACHE PATH "Shared library output directory")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MORTRED_BIN_OUTPUT_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${MORTRED_LIB_OUTPUT_DIR})

# Vendored deps: explicit imported targets (exact soname + fatal error if missing + auto-reconfigure on change)
include(${PROJECT_ROOT_DIR}/cmake/VendoredDeps.cmake)
mortred_probe_workflow()

# CI / unit-test mode: skip CUDA/MNN/TRT-dependent models and servers; build only common + tests
option(MORTRED_BUILD_FULL "Build full project (models/server/apps, requires CUDA/MNN/TRT)" ON)
# Deployment profile: gpu = full CUDA/TensorRT stack (default, unchanged);
# cpu = MNN-CPU + ORT-CPU backends only, no TensorRT (the trt backend is
# compiled out; trt-configured models fail init with a clear error). Single
# source of truth for the deployment story - see docs/deployment.md.
set(MORTRED_BUILD_PROFILE "gpu" CACHE STRING "Deployment profile: gpu | cpu")
set_property(CACHE MORTRED_BUILD_PROFILE PROPERTY STRINGS gpu cpu)
option(MORTRED_INSTALL "Create the tree-style install target (bin/lib tree)" OFF)

# CI quality gate: treat compiler warnings as errors (enabled in tests-only builds)
option(MORTRED_ENABLE_WERROR "Treat compiler warnings as errors (used by CI quality gate)" OFF)
if(MORTRED_ENABLE_WERROR)
    add_compile_options(-Wall -Wextra -Werror)
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
    # exact versions, fail if missing; the cpu profile drops the CUDA/TRT set
    mortred_import_shared(mnn            libMNN.so)
    mortred_import_shared(fmt            libfmt.so.9)
    mortred_import_shared(onnxruntime    libonnxruntime.so.1.18.0)
    # the gateway calls ApiKeyManager::authenticate (SHA-256); pin the vendored
    # crypto so link-time and runtime (LD_LIBRARY_PATH install tree) agree
    mortred_import_shared(crypto         libcrypto.so.1.1)
    if(MORTRED_BUILD_PROFILE STREQUAL "cpu")
        message(STATUS "Build profile: cpu (MNN-CPU + ORT-CPU, TensorRT compiled out)")
    else()
        message(STATUS "Build profile: gpu (full CUDA/TensorRT stack)")
        find_package(CUDAToolkit REQUIRED)
        mortred_import_shared(mnn_cuda       libMNN_Cuda_Main.so)
        mortred_import_shared(nvinfer        libnvinfer.so.8)
        mortred_import_shared(nvinfer_plugin libnvinfer_plugin.so.8)
        mortred_import_shared(nvonnxparser   libnvonnxparser.so.8)
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
    install(DIRECTORY ${PROJECT_ROOT_DIR}/deploy/caddy DESTINATION deploy)
    install(DIRECTORY scripts DESTINATION .
            FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                             GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    install(DIRECTORY src/control/supervisor/ui DESTINATION share/mortred)
endif()
