# build models libs

set(MODELS_LIB_SRC
    ${CMAKE_CURRENT_LIST_DIR}/base_model.h
    ${CMAKE_CURRENT_LIST_DIR}/backend/backend_config.h
    ${CMAKE_CURRENT_LIST_DIR}/backend/backend_cv_model.h
    ${CMAKE_CURRENT_LIST_DIR}/backend/f32_output.h
    ${CMAKE_CURRENT_LIST_DIR}/backend/inference_context.h
    ${CMAKE_CURRENT_LIST_DIR}/backend/mnn_session.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/mnn_session.h
    ${CMAKE_CURRENT_LIST_DIR}/backend/model_runtime.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/model_runtime.h
    ${CMAKE_CURRENT_LIST_DIR}/backend/multi_session_model.h
    ${CMAKE_CURRENT_LIST_DIR}/backend/param_spec.h
    ${CMAKE_CURRENT_LIST_DIR}/backend/ort_session.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/ort_session.h
    ${CMAKE_CURRENT_LIST_DIR}/backend/session.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/session.h
    ${CMAKE_CURRENT_LIST_DIR}/backend/request_geometry.h
    ${CMAKE_CURRENT_LIST_DIR}/backend/tensor.h
    ${CMAKE_CURRENT_LIST_DIR}/backend/tensor_contract.h
    ${CMAKE_CURRENT_LIST_DIR}/catalog/model_entry.h
    ${CMAKE_CURRENT_LIST_DIR}/cv_image_input.h
    ${CMAKE_CURRENT_LIST_DIR}/model_io_define.h
    ${CMAKE_CURRENT_LIST_DIR}/clip/simple_tokenizer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mot/byte_tracker/byte_tracker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mot/byte_tracker/kalman_filter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mot/byte_tracker/lapjv.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mot/byte_tracker/strack.cpp
    ${CMAKE_CURRENT_LIST_DIR}/segment_anything/sam_automask_generator/sam_amg_decoder.cpp
    ${CMAKE_CURRENT_LIST_DIR}/segment_anything/sam_prediction/sam_prompt_decoder.cpp
    ${CMAKE_CURRENT_LIST_DIR}/segment_anything/sam_prediction/sam_vit_encoder.cpp
)
add_library(models SHARED ${MODELS_LIB_SRC})
# lapjv.cpp is a vendored third-party LAPJV solver; its pre-existing warnings skip the -Werror gate
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
)
if(MORTRED_BUILD_PROFILE STREQUAL "cpu")
    # cpu profile: the TensorRT session is compiled out entirely; the backend
    # factory (session.cpp) returns a clear init error for trt configs. TRT
    # headers/libs are absent in a cpu dependency tree, so they must not be
    # referenced here at all.
    # NOTE: the cpu branch deliberately does NOT define MORTRED_HAS_TRT, so
    # `#ifdef MORTRED_HAS_TRT` in session.cpp is false; defining it to 0 would
    # make #ifdef true (classic trap).
    target_link_libraries(
        models
        PUBLIC
        common
        vendored::mnn
        vendored::workflow
        vendored::fmt
        vendored::onnxruntime
        ${OpenCV_LIBS}
        glog::glog
    )
else()
    # TensorRT header dir switches with version (TensorRT-8.6.1.6 / TensorRT-10.x); new code always
    # uses version-less official paths such as #include "NvInfer.h"
    file(GLOB MORTRED_TRT_INCLUDE_DIRS "${PROJECT_ROOT_DIR}/3rd_party/include/TensorRT-*")
    list(SORT MORTRED_TRT_INCLUDE_DIRS)
    list(GET MORTRED_TRT_INCLUDE_DIRS 0 MORTRED_TRT_INCLUDE_DIR)
    if(NOT MORTRED_TRT_INCLUDE_DIR)
        message(FATAL_ERROR "no vendored TensorRT include dir found under 3rd_party/include/TensorRT-*")
    endif()
    message(STATUS "vendored tensorrt headers: ${MORTRED_TRT_INCLUDE_DIR}")
    target_include_directories(models SYSTEM PUBLIC ${MORTRED_TRT_INCLUDE_DIR})
    target_compile_definitions(models PUBLIC MORTRED_HAS_TRT=1)
    # the gpu profile compiles the TensorRT session into the models lib (the
    # cpu profile above leaves it out entirely)
    target_sources(
        models
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/backend/trt_session.cpp
        ${CMAKE_CURRENT_LIST_DIR}/backend/trt_session.h
    )
    target_link_libraries(
        models
        PUBLIC
        common
        vendored::mnn
        vendored::mnn_cuda
        vendored::workflow
        vendored::fmt
        vendored::onnxruntime
        vendored::nvinfer
        vendored::nvinfer_plugin
        vendored::nvonnxparser
        CUDA::cudart
        ${OpenCV_LIBS}
        glog::glog
    )
endif()
# Deprecation suppression applies only to targets needing old OpenCV APIs (replaces the old global suppression)
target_compile_options(models PRIVATE -Wno-deprecated-declarations)
set_target_properties(models PROPERTIES LINKER_LANGUAGE CXX)
