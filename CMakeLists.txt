cmake_minimum_required(VERSION 3.15)

project(mortred_model_server LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# 兼容既有代码对 deprecated OpenCV API 的大量使用；警告即错误开关由 CI 质量门禁单独控制
add_compile_options(-Wno-deprecated-declarations)

set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})

# 构建类型：仅在未指定时默认 Release，不再强制覆盖用户的 Debug/RelWithDebInfo 选择
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug/Release/RelWithDebInfo)" FORCE)
endif()

message(STATUS ">>>>>>>>>>>>>")
message(STATUS "Mortred Model Server INFO:")
message(STATUS "\tSystem: ${CMAKE_SYSTEM_NAME}")
message(STATUS "\tProcessor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "\tBuild Path: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "\tBuild Type: ${CMAKE_BUILD_TYPE}")

# 输出目录可配置，默认保持历史路径 _bin/_lib（web_console/ServerManager 依赖该布局）
set(MORTRED_BIN_OUTPUT_DIR ${PROJECT_ROOT_DIR}/_bin CACHE PATH "Executable output directory")
set(MORTRED_LIB_OUTPUT_DIR ${PROJECT_ROOT_DIR}/_lib CACHE PATH "Shared library output directory")
set(EXECUTABLE_OUTPUT_PATH ${MORTRED_BIN_OUTPUT_DIR})
set(LIBRARY_OUTPUT_PATH ${MORTRED_LIB_OUTPUT_DIR})
file(GLOB MNN_LIBS ${PROJECT_ROOT_DIR}/3rd_party/libs/*MNN*so*)
file(GLOB_RECURSE WORKFLOW_LIBS ${PROJECT_ROOT_DIR}/3rd_party/libs/*workflow*so)
file(GLOB_RECURSE FMT_LIBRARIES ${PROJECT_ROOT_DIR}/3rd_party/libs/*fmt*so)
file(GLOB_RECURSE ONNXRUNTIME_LIBRARIES ${PROJECT_ROOT_DIR}/3rd_party/libs/*onnxruntime*so)
file(GLOB_RECURSE NVINFER_LIBRARIES ${PROJECT_ROOT_DIR}/3rd_party/libs/*nvinfer*.so*)
file(GLOB_RECURSE TRT_ONNXPARSER_LIBRARIES ${PROJECT_ROOT_DIR}/3rd_party/libs/*nvonnxparser*.so*)

# CI / 单测模式：跳过依赖 CUDA/MNN/TRT 的模型与服务，只构建 common + tests
option(MORTRED_BUILD_FULL "Build full project (models/server/apps, requires CUDA/MNN/TRT)" ON)

# CI 质量门禁：将编译器警告视为错误（tests-only 构建下启用）
option(MORTRED_ENABLE_WERROR "Treat compiler warnings as errors (used by CI quality gate)" OFF)
if(MORTRED_ENABLE_WERROR)
    add_compile_options(-Wall -Wextra -Werror)
endif()

# find package
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)
find_package(Eigen3 REQUIRED)

# 公共 include 作用域：由各库目标以 PUBLIC 形式向外传播，替代全局 include_directories。
# 3rd_party 与系统依赖标记为 SYSTEM：vendored 头文件的警告不参与 -Werror 质量门禁。
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

if(MORTRED_BUILD_FULL)
    find_package(CUDA REQUIRED)
    add_subdirectory(src/models)
    add_subdirectory(src/factory)
    add_subdirectory(src/server)
    add_subdirectory(src/apps)
endif()

# add unittest
enable_testing()
add_subdirectory(test EXCLUDE_FROM_ALL)
