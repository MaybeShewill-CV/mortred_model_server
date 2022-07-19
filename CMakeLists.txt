cmake_minimum_required(VERSION 3.15)

project(mortred_model_server)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
set(CMAKE_CXX_FLAGS "-std=c++14")
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

# find package
find_package(Threads REQUIRED)
if(CMAKE_HOST_SYSTEM_NAME MATCHES "Windows")
    set(OpenCV_DIR "E:\\opencv-4.5.0\\out\\build\\x64-Release")
endif()
find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)

include_directories(${GLOG_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

# add subdirectory
add_subdirectory(src/common)
add_subdirectory(src/models)
add_subdirectory(src/factory)
add_subdirectory(src/server)
add_subdirectory(src/apps)

# add unittest
# add_subdirectory(test EXCLUDE_FROM_ALL)
