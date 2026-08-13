cmake_minimum_required(VERSION 3.15)

set(CMAKE_BUILD_TYPE Release)

project(mm_ai_server_test LANGUAGES C CXX)

find_program(CMAKE_MEMORYCHECK_COMMAND valgrind)
set(memcheck_command ${CMAKE_MEMORYCHECK_COMMAND} ${CMAKE_MEMORYCHECK_COMMAND_OPTIONS} --error-exitcode=1 --leak-check=full)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})

enable_testing()
find_package(GTest REQUIRED)

# GTest 目标名在模块模式与配置模式下不同，做兼容选择
if(TARGET GTest::GTest)
    set(GTEST_LIBS GTest::GTest GTest::Main)
elseif(TARGET GTest::gtest)
    set(GTEST_LIBS GTest::gtest GTest::gtest_main)
else()
    set(GTEST_LIBS ${GTEST_LIBRARIES})
endif()

if (WIN32)
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   /MP /wd4200")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /wd4200 /std:c++14")
else ()
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -Wall -fPIC -pipe -std=gnu90")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -pipe -std=c++17")
endif ()

include_directories(${PROJECT_ROOT_DIR}/src)
include_directories(${PROJECT_ROOT_DIR}/3rd_party/include)

set(TEST_LIST
    base64_unittest
    md5_unittest
    file_path_util_unittest
    json_request_parser_unittest
    blocking_worker_queue_unittest
    status_code_unittest
    llm_request_parser_unittest
    time_stamp_unittest
    cv_utils_unittest
    llm_datatype_unittest
    model_factory_unittest
    byte_tracker_unittest
    simple_tokenizer_unittest
    request_size_limit_unittest
    web_console_auth_unittest
    rate_limiter_unittest
    cv_image_input_unittest
    worker_nums_unittest
)

if(MORTRED_BUILD_FULL)
    list(APPEND TEST_LIST model_golden_test)
endif()

foreach(src ${TEST_LIST})
    set(EXTRA_SRCS "")
    set(EXTRA_LIBS "")
    if(${src} STREQUAL "byte_tracker_unittest")
        set(EXTRA_SRCS
            ${PROJECT_ROOT_DIR}/src/models/mot/byte_tracker/byte_tracker.cpp
            ${PROJECT_ROOT_DIR}/src/models/mot/byte_tracker/kalman_filter.cpp
            ${PROJECT_ROOT_DIR}/src/models/mot/byte_tracker/lapjv.cpp
            ${PROJECT_ROOT_DIR}/src/models/mot/byte_tracker/strack.cpp)
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "cv_utils_unittest")
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "cv_image_input_unittest")
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "worker_nums_unittest")
        set(EXTRA_LIBS glog::glog)
    elseif(${src} STREQUAL "model_factory_unittest")
        set(EXTRA_LIBS glog::glog)
    elseif(${src} STREQUAL "model_golden_test")
        set(EXTRA_LIBS models glog::glog ${OpenCV_LIBS} ${MNN_LIBS})
    elseif(${src} STREQUAL "simple_tokenizer_unittest")
        set(EXTRA_SRCS ${PROJECT_ROOT_DIR}/src/models/clip/simple_tokenizer.cpp)
        set(EXTRA_LIBS glog::glog ${OpenCV_LIBS})
    elseif(${src} STREQUAL "web_console_auth_unittest")
        set(EXTRA_SRCS ${PROJECT_ROOT_DIR}/src/apps/web_console/backend/auth.cpp)
    endif()
    add_executable(${src} EXCLUDE_FROM_ALL ${src}.cc ${EXTRA_SRCS})
    target_link_libraries(${src} common ${EXTRA_LIBS} ${GTEST_LIBS})
    add_test(NAME ${src} COMMAND $<TARGET_FILE:${src}> WORKING_DIRECTORY ${PROJECT_ROOT_DIR})
    add_dependencies(check ${src})
endforeach()

if(CMAKE_MEMORYCHECK_COMMAND)
    foreach(src ${TEST_LIST})
        add_test(${src}-memory-check ${memcheck_command} ./${src})
    endforeach()
endif()
