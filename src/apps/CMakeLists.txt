# build apps

include_directories(
    ${PROJECT_ROOT_DIR}/3rd_party/include
    ${PROJECT_ROOT_DIR}/src
)

# build benchmark apps
set(MODEL_BENCHMARK_LIST
    classification
    enhancement
    feature_point
    object_detection
    ocr
    segmentation
    matting
    sam
)

foreach(dir_name ${MODEL_BENCHMARK_LIST})
    file(GLOB_RECURSE src_list ./model_benchmark/${dir_name}/*)
    foreach(src ${src_list})
        string(REGEX REPLACE ".+/(.+)\\..*" "\\1" file_name ${src})
        add_executable(${file_name}.out ${src})
        target_link_libraries(${file_name}.out common models factory ${WORKFLOW_LIBS} ${GLOG_LIBRARIES})
    endforeach()
endforeach()

# build server app
set(MODEL_SERVER_APP_LIST
    classification
    enhancement
    feature_point
    object_detection
    ocr
    proxy
    scene_segmentation
    matting
)

foreach(dir_name ${MODEL_SERVER_APP_LIST})
    file(GLOB_RECURSE src_list ./server/${dir_name}/*)
    foreach(src ${src_list})
        string(REGEX REPLACE ".+/(.+)\\..*" "\\1" file_name ${src})
        add_executable(${file_name}.out ${src})
        target_link_libraries(${file_name}.out common models factory ${WORKFLOW_LIBS} ${GLOG_LIBRARIES})
    endforeach()
endforeach()
