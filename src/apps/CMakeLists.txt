# build apps

include_directories(
    ${PROJECT_ROOT_DIR}/3rd_party/include
    ${PROJECT_ROOT_DIR}/src
)

file(GLOB_RECURSE GLOG_LIBS ${PROJECT_ROOT_DIR}/3rd_party/libs/*glog*so)
file(GLOB_RECURSE WORKFLOW_LIBS ${PROJECT_ROOT_DIR}/3rd_party/libs/*workflow*so)

# build benchmark apps
set(MODEL_BENCHMARK_LIST
    classification
    enhancement
    feature_point
    object_detection
    ocr
    segmentation
)

foreach(dir_name ${MODEL_BENCHMARK_LIST})
    file(GLOB_RECURSE src_list ./model_benchmark/${dir_name}/*)
    foreach(src ${src_list})
        string(REGEX REPLACE ".+/(.+)\\..*" "\\1" file_name ${src})
        add_executable(${file_name}.out ${src})
        target_link_libraries(${file_name}.out common models factory ${WORKFLOW_LIBS} ${GLOG_LIBS})
    endforeach()
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
    classification
    feature_point
    object_detection
    proxy
)

foreach(dir_name ${MODEL_SERVER_APP_LIST})
    file(GLOB_RECURSE src_list ./server/${dir_name}/*)
    foreach(src ${src_list})
        string(REGEX REPLACE ".+/(.+)\\..*" "\\1" file_name ${src})
        add_executable(${file_name}.out ${src})
        target_link_libraries(${file_name}.out common models factory ${WORKFLOW_LIBS} ${GLOG_LIBS})
    endforeach()
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
