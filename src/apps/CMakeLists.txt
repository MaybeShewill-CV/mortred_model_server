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
    ${WORKFLOW_LIBS}
    ${GLOG_LIBS}
)

# add dbtext test tools
add_executable(dbtext_test.out ./dbtext_test.cpp)
target_link_libraries(
    dbtext_test.out
    common
    models
    ${GLOG_LIBS}
)

# add yolov5 test tools
add_executable(yolov5_test.out ./yolov5_test.cpp)
target_link_libraries(
    yolov5_test.out
    common
    models
    ${GLOG_LIBS}
)

# add bisenetv2 test tools
add_executable(bisenetv2_test.out ./bisenetv2_test.cpp)
target_link_libraries(
    bisenetv2_test.out
    common
    models
    ${GLOG_LIBS}
)