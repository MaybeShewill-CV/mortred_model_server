# build models libs

set(MODELS_LIB_SRC
    ${CMAKE_CURRENT_LIST_DIR}/base_model.h
    ${CMAKE_CURRENT_LIST_DIR}/cv_image_input.h
    ${CMAKE_CURRENT_LIST_DIR}/mnn_helper.h
    ${CMAKE_CURRENT_LIST_DIR}/model_io_define.h
    ${CMAKE_CURRENT_LIST_DIR}/clip/openai_clip_text_encoder.cpp
    ${CMAKE_CURRENT_LIST_DIR}/clip/openai_clip_vit_encoder.cpp
    ${CMAKE_CURRENT_LIST_DIR}/clip/simple_tokenizer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mot/byte_tracker/byte_tracker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mot/byte_tracker/kalman_filter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mot/byte_tracker/lapjv.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mot/byte_tracker/strack.cpp
    ${CMAKE_CURRENT_LIST_DIR}/segment_anything/sam_automask_generator/sam_amg_decoder.cpp
    ${CMAKE_CURRENT_LIST_DIR}/segment_anything/sam_prediction/sam_prompt_decoder.cpp
    ${CMAKE_CURRENT_LIST_DIR}/segment_anything/sam_prediction/sam_vit_encoder.cpp
    ${CMAKE_CURRENT_LIST_DIR}/trt_helper/onnx2trt_model_builder.cpp
    ${CMAKE_CURRENT_LIST_DIR}/trt_helper/trt_helper.cpp
)
add_library(models SHARED ${MODELS_LIB_SRC})
# lapjv.cpp 为 vendored 第三方 LAPJV 求解器，其存量警告不参与 -Werror 门禁
set_source_files_properties(
    ${CMAKE_CURRENT_LIST_DIR}/mot/byte_tracker/lapjv.cpp
    PROPERTIES COMPILE_OPTIONS "-Wno-unused-parameter"
)
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
