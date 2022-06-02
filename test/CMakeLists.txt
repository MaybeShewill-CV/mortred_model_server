cmake_minimum_required(VERSION 3.15)

set(CMAKE_BUILD_TYPE Release)

project(mm_ai_server_test LANGUAGES C CXX)

find_program(CMAKE_MEMORYCHECK_COMMAND valgrind)
set(memcheck_command ${CMAKE_MEMORYCHECK_COMMAND} ${CMAKE_MEMORYCHECK_COMMAND_OPTIONS} --error-exitcode=1 --leak-check=full)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})

enable_testing()
find_package(GTest REQUIRED)

if (WIN32)
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   /MP /wd4200")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /wd4200 /std:c++14")
else ()
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -Wall -fPIC -pipe -std=gnu90")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -pipe -std=c++11 -fno-exceptions")
endif ()

include_directories(${PROJECT_ROOT_DIR}/src)

set(TEST_LIST
    base64_unittest
)

foreach(src ${TEST_LIST})
    add_executable(${src} EXCLUDE_FROM_ALL ${src}.cc)
    target_link_libraries(${src} common GTest::GTest GTest::Main)
    add_test(${src} ${src})
    add_dependencies(check ${src})
endforeach()

foreach(src ${TEST_LIST})
    add_test(${src}-memory-check ${memcheck_command} ./${src})
endforeach()