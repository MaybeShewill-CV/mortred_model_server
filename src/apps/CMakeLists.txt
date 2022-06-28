# build apps

include_directories(
    ${PROJECT_ROOT_DIR}/3rd_party/include
    ${PROJECT_ROOT_DIR}/src
)

file(GLOB_RECURSE GLOG_LIBS ${PROJECT_ROOT_DIR}/3rd_party/libs/*glog*so)
file(GLOB_RECURSE WORKFLOW_LIBS ${PROJECT_ROOT_DIR}/3rd_party/libs/*workflow*so)

# build benchmark apps
set(MODEL_BENCHMARK_LIST
    yolov5_benchmark
    nanodet_benchmark
    bisenetv2_benchmark
    dbnet_benchmark
    enlightengan_benchmark
    mobilenetv2_benchmark
    resnet_benchmark
    densenet_benchmark
    attentivegan_benchmark
    superpoint_benchmark
    libface_benchmark
)

foreach(src ${MODEL_BENCHMARK_LIST})
    add_executable(${src}.out ./model_benchmark/${src}.cpp)
    target_link_libraries(${src}.out common models factory ${WORKFLOW_LIBS} ${GLOG_LIBS})
endforeach()

# build fast batch process app
set(MODEL_FAST_BATCH_PROCESS_LIST
    enlightengan_fast_batch_process
)

foreach(src ${MODEL_FAST_BATCH_PROCESS_LIST})
    add_executable(${src}.out ./fast_batch_process/${src}.cpp)
    target_link_libraries(${src}.out common models factory ${WORKFLOW_LIBS} ${GLOG_LIBS})
endforeach()

# build server app
set(MODEL_SERVER_APP_LIST
    resnet_classification_server
    nanodet_detection_server
    yolov5_detection_server
    mobilenetv2_classification_server
    libface_detection_server
    superpoint_fp_det_server
)

foreach(src ${MODEL_SERVER_APP_LIST})
    add_executable(${src}.out ./server/${src}.cpp)
    target_link_libraries(${src}.out common models factory server ${WORKFLOW_LIBS} ${GLOG_LIBS})
endforeach()

# add test tools
add_executable(test.out ./test.cpp)
target_link_libraries(
    test.out
    common
    models
    factory
    ${WORKFLOW_LIBS}
    ${GLOG_LIBS}
)
