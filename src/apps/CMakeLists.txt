# build apps

include_directories(
    ${PROJECT_ROOT_DIR}/3rd_party/include
    ${PROJECT_ROOT_DIR}/src
)

file(GLOB_RECURSE BOOST_LIBS ${PROJECT_ROOT_DIR}/3rd_party/libs/*boost*so)
file(GLOB_RECURSE OPENCV_LIBS ${PROJECT_ROOT_DIR}/3rd_party/libs/*opencv*so)
file(GLOB_RECURSE GLOG_LIBS ${PROJECT_ROOT_DIR}/3rd_party/libs/*glog*so)
file(GLOB_RECURSE WORKFLOW_LIBS ${PROJECT_ROOT_DIR}/3rd_party/libs/*workflow*so)

# add test tools
add_executable(test.out ./test.cpp)
target_link_libraries(
    test.out
    common
)