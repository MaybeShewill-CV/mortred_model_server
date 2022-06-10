# build apps

include_directories(
    ${PROJECT_ROOT_DIR}/3rd_party/include
    ${PROJECT_ROOT_DIR}/src
)

file(GLOB_RECURSE BOOST_LIBS ${PROJECT_ROOT_DIR}/3rd_party/libs/*boost*so)
file(GLOB_RECURSE OPENCV_LIBS ${PROJECT_ROOT_DIR}/3rd_party/libs/*opencv*so)
file(GLOB_RECURSE GLOG_LIBS ${PROJECT_ROOT_DIR}/3rd_party/libs/*glog*so)
file(GLOB_RECURSE WORKFLOW_LIBS ${PROJECT_ROOT_DIR}/3rd_party/libs/*workflow*so)

set(MODEL_BENCHMARK_LIST
    yolov5_benchmark
    nanodet_benchmark
    bisenetv2_benchmark
    dbnet_benchmark
)

foreach(src ${MODEL_BENCHMARK_LIST})
    add_executable(${src}.out ./model_benchmark/${src}.cpp)
    target_link_libraries(${src}.out common models factory ${WORKFLOW_LIBS} ${GLOG_LIBS})
endforeach()

# add test tools
add_executable(test.out ./test.cpp)
target_link_libraries(
    test.out
    common
    ${WORKFLOW_LIBS}
    ${GLOG_LIBS}
)
