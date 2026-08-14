# build models libs

file(GLOB_RECURSE MODELS_LIB_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/*)
add_library(models SHARED ${MODELS_LIB_SRC})
target_include_directories(
    models PUBLIC
    ${PROJECT_ROOT_DIR}/src
)
target_include_directories(
    models SYSTEM PUBLIC
    ${PROJECT_ROOT_DIR}/3rd_party/include
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(
    models
    common
    ${MNN_LIBS}
    ${OpenCV_LIBS}
    glog::glog
    ${ONNXRUNTIME_LIBRARIES}
    ${FMT_LIBRARIES}
    ${NVINFER_LIBRARIES}
    ${TRT_ONNXPARSER_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${WORKFLOW_LIBS}
)
set_target_properties(models PROPERTIES LINKER_LANGUAGE CXX)
