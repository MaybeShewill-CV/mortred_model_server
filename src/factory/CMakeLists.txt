# build factory libs

set(FACTORY_LIB_SRC
    ${CMAKE_CURRENT_LIST_DIR}/base_factory.h
    ${CMAKE_CURRENT_LIST_DIR}/classification_task.h
    ${CMAKE_CURRENT_LIST_DIR}/clip_task.h
    ${CMAKE_CURRENT_LIST_DIR}/cv_catalog.h
    ${CMAKE_CURRENT_LIST_DIR}/diffusion_model_adapter.h
    ${CMAKE_CURRENT_LIST_DIR}/diffusion_task.h
    ${CMAKE_CURRENT_LIST_DIR}/enhancement_task.h
    ${CMAKE_CURRENT_LIST_DIR}/feature_embedding_task.h
    ${CMAKE_CURRENT_LIST_DIR}/feature_point_task.h
    ${CMAKE_CURRENT_LIST_DIR}/matting_task.h
    ${CMAKE_CURRENT_LIST_DIR}/model_catalog.h
    ${CMAKE_CURRENT_LIST_DIR}/mono_depth_estimate_task.h
    ${CMAKE_CURRENT_LIST_DIR}/obj_detection_task.h
    ${CMAKE_CURRENT_LIST_DIR}/ocr_task.h
    ${CMAKE_CURRENT_LIST_DIR}/sam_task.h
    ${CMAKE_CURRENT_LIST_DIR}/scene_segmentation_task.h
)
add_library(factory SHARED ${FACTORY_LIB_SRC})
target_include_directories(
    factory PUBLIC
    ${PROJECT_ROOT_DIR}/src
)
target_include_directories(
    factory SYSTEM PUBLIC
    ${PROJECT_ROOT_DIR}/3rd_party/include
)
target_link_libraries(
    factory
    common
    models
    server
)
set_target_properties(factory PROPERTIES LINKER_LANGUAGE CXX)
