cmake_minimum_required(VERSION 3.18)

project(mortred_model_server LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})

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
    find_package(CUDAToolkit REQUIRED)
    # CUDA inference stack (MNN-TRT backend / onnxruntime / TensorRT): exact versions, fail if missing
    mortred_import_shared(mnn            libMNN.so)
    mortred_import_shared(mnn_cuda       libMNN_Cuda_Main.so)
    mortred_import_shared(fmt            libfmt.so.9)
    mortred_import_shared(onnxruntime    libonnxruntime.so.1.18.0)
    mortred_import_shared(nvinfer        libnvinfer.so.8)
    mortred_import_shared(nvinfer_plugin libnvinfer_plugin.so.8)
    mortred_import_shared(nvonnxparser   libnvonnxparser.so.8)
    add_subdirectory(src/models)
    add_subdirectory(src/factory)
    add_subdirectory(src/server)
    add_subdirectory(src/apps)
endif()

# add unittest
enable_testing()
add_subdirectory(test EXCLUDE_FROM_ALL)

# Install target (enabled with MORTRED_INSTALL=ON): tree-style install of bin/lib.
# Runtime entry (Docker/systemd) sets LD_LIBRARY_PATH=/opt/mortred/lib explicitly;
# do not rely on the build's RUNPATH, which may contain absolute source/build-machine paths.
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
    install(DIRECTORY scripts DESTINATION .
            FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                             GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    install(DIRECTORY src/control/supervisor/ui DESTINATION share/mortred)
endif()
