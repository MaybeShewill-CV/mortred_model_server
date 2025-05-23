# build models libs

include_directories(
    ${PROJECT_ROOT_DIR}/3rd_party/include
    ${PROJECT_ROOT_DIR}/src
)

file(GLOB_RECURSE MODELS_LIB_SRC ${CMAKE_CURRENT_LIST_DIR}/*)
add_library(models SHARED ${MODELS_LIB_SRC})
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
    ${LLAMA_LIBRARIES}
    ${GGML_LIBRARIES}
    ${FAISS_LIBRARIES}
)
set_target_properties(models PROPERTIES LINKER_LANGUAGE CXX)
