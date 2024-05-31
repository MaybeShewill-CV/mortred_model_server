cmake_minimum_required(VERSION 3.15)

project(mortred_model_server)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wno-deprecated-declarations")
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})

# target options
set(CMAKE_BUILD_TYPE Release)
IF(CMAKE_BUILD_TYPE MATCHES Debug)
    message("Build type debug")
ELSE()
    message("Build type release")
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

message(STATUS ">>>>>>>>>>>>>")
message(STATUS "Mortred Model Server INFO:")
message(STATUS "\tSystem: ${CMAKE_SYSTEM_NAME}")
message(STATUS "\tProcessor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "\tBuild Path: ${CMAKE_CURRENT_BINARY_DIR}")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_ROOT_DIR}/_bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_LIST_DIR}/_lib)
file(GLOB MNN_LIBS ${PROJECT_ROOT_DIR}/3rd_party/libs/*MNN*so*)
file(GLOB_RECURSE WORKFLOW_LIBS ${PROJECT_ROOT_DIR}/3rd_party/libs/*workflow*so)
file(GLOB_RECURSE GLOG_LIBRARIES ${PROJECT_ROOT_DIR}/3rd_party/libs/*glog*so)
file(GLOB_RECURSE FMT_LIBRARIES ${PROJECT_ROOT_DIR}/3rd_party/libs/*fmt*so)
file(GLOB_RECURSE ONNXRUNTIME_LIBRARIES ${PROJECT_ROOT_DIR}/3rd_party/libs/*onnxruntime*so)
file(GLOB_RECURSE NVINFER_LIBRARIES ${PROJECT_ROOT_DIR}/3rd_party/libs/*nvinfer*.so*)
file(GLOB_RECURSE TRT_ONNXPARSER_LIBRARIES ${PROJECT_ROOT_DIR}/3rd_party/libs/*nvonnxparser*.so*)

# find package
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${GLOG_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

# add subdirectory
add_subdirectory(src/common)
add_subdirectory(src/models)
add_subdirectory(src/factory)
add_subdirectory(src/server)
add_subdirectory(src/apps)

# add unittest
# add_subdirectory(test EXCLUDE_FROM_ALL)
