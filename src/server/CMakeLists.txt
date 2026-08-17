# build server libs

set(SERVER_LIB_SRC
    ${CMAKE_CURRENT_LIST_DIR}/abstract_server.h
    ${CMAKE_CURRENT_LIST_DIR}/base_server_impl.h
    ${CMAKE_CURRENT_LIST_DIR}/http_status.h
    ${CMAKE_CURRENT_LIST_DIR}/rate_limiter.h
    ${CMAKE_CURRENT_LIST_DIR}/classification/densenet_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/classification/mobilenetv2_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/classification/resnet_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/enhancement/attentive_gan_derain_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/enhancement/enlighten_gan_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/enhancement/real_esr_gan_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/feature_point/superpoint_fp_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/matting/modnet_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/matting/pp_matting_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mono_depth_estimation/depth_anything_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mono_depth_estimation/metric3d_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/object_detection/centerface_det_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/object_detection/libface_det_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/object_detection/nano_det_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/object_detection/yolov5_det_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/object_detection/yolov6_det_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/object_detection/yolov7_det_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/object_detection/yolov8_det_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ocr/dbnet_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/scene_segmentation/bisenetv2_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/scene_segmentation/hrnet_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/scene_segmentation/pphuman_seg_server.cpp
)
add_library(server SHARED ${SERVER_LIB_SRC})
target_include_directories(
    server PUBLIC
    ${PROJECT_ROOT_DIR}/src
)
target_include_directories(
    server SYSTEM PUBLIC
    ${PROJECT_ROOT_DIR}/3rd_party/include
)
target_link_libraries(
    server
    common
    models
    ${WORKFLOW_LIBS}
    glog::glog
)
set_target_properties(server PROPERTIES LINKER_LANGUAGE CXX)
