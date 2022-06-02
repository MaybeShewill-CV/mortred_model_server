# build apps

include_directories(
    ${PROJECT_ROOT_DIR}/3rd_party/include
    ${PROJECT_ROOT_DIR}/src
)

file(GLOB_RECURSE BOOST_LIBS ${PROJECT_ROOT_DIR}/3rd_party/libs/*boost*so)
file(GLOB_RECURSE OPENCV_LIBS ${PROJECT_ROOT_DIR}/3rd_party/libs/*opencv*so)
file(GLOB_RECURSE GLOG_LIBS ${PROJECT_ROOT_DIR}/3rd_party/libs/*glog*so)
file(GLOB_RECURSE WORKFLOW_LIBS ${PROJECT_ROOT_DIR}/3rd_party/libs/*workflow*so)

# add yolov5 detection tools
add_executable(yolov5_detector.out ./detection_tools/yolov5_detector.cpp)
target_link_libraries(
    yolov5_detector.out
    common
    detector
    interface
    ${BOOST_LIBS}
    ${OPENCV_LIBS}
    ${GLOG_LIBS}
    ${WORKFLOW_LIBS}
)

# add nano detection tools
add_executable(nano_detector.out ./detection_tools/nano_detector.cpp)
target_link_libraries(
    nano_detector.out
    common
    detector
    interface
    ${BOOST_LIBS}
    ${OPENCV_LIBS}
    ${GLOG_LIBS}
    ${WORKFLOW_LIBS}
)

# add privacy detection tools
add_executable(privacy_detector.out ./detection_tools/privacy_detector.cpp)
target_link_libraries(
    privacy_detector.out
    common
    detector
    interface
    ${BOOST_LIBS}
    ${OPENCV_LIBS}
    ${GLOG_LIBS}
    ${WORKFLOW_LIBS}
)

# add privacy detection tools
add_executable(db_text_detector.out ./detection_tools/text_detector.cpp)
target_link_libraries(
    db_text_detector.out
    common
    detector
    interface
    ${BOOST_LIBS}
    ${OPENCV_LIBS}
    ${GLOG_LIBS}
    ${WORKFLOW_LIBS}
)

# add shang tang privacy pre task server
add_executable(st_privacy_pre_task_server.out ./privacy_server/st_privacy_pre_task_service.cpp)
target_link_libraries(
    st_privacy_pre_task_server.out
    common
    detector
    interface
    server
    ${BOOST_LIBS}
    ${OPENCV_LIBS}
    ${GLOG_LIBS}
    ${WORKFLOW_LIBS}
)

# add shang tang privacy blur task server
add_executable(st_privacy_blur_task_server.out ./privacy_server/st_privacy_blur_task_service.cpp)
target_link_libraries(
    st_privacy_blur_task_server.out
    common
    detector
    interface
    server
    ${BOOST_LIBS}
    ${OPENCV_LIBS}
    ${GLOG_LIBS}
    ${WORKFLOW_LIBS}
)

# add tesla privacy pre task server
add_executable(tesla_privacy_pre_task_server.out ./privacy_server/tesla_privacy_pre_task_service.cpp)
target_link_libraries(
    tesla_privacy_pre_task_server.out
    common
    detector
    interface
    server
    ${BOOST_LIBS}
    ${OPENCV_LIBS}
    ${GLOG_LIBS}
    ${WORKFLOW_LIBS}
)

# add video decode tools
add_executable(decode_video.out tmp_tools/decode_h264_video_frame.cpp)
target_link_libraries(
    decode_video.out
    common
    detector
    interface
    ${BOOST_LIBS}
    ${OPENCV_LIBS}
    ${GLOG_LIBS}
    ${WORKFLOW_LIBS}
)

# add proxy server
add_executable(api_proxy_server.out web_server/api_proxy_server.cpp)
target_link_libraries(
    api_proxy_server.out
    common
    detector
    interface
    server
    ${BOOST_LIBS}
    ${OPENCV_LIBS}
    ${GLOG_LIBS}
    ${WORKFLOW_LIBS}
)

# add image file server
add_executable(image_file_server.out web_server/image_file_server.cpp)
target_link_libraries(
    image_file_server.out
    common
    detector
    interface
    server
    ${BOOST_LIBS}
    ${OPENCV_LIBS}
    ${GLOG_LIBS}
    ${WORKFLOW_LIBS}
)

# add db text detection server
add_executable(db_text_detection_server.out web_server/db_text_detection_server.cpp)
target_link_libraries(
    db_text_detection_server.out
    common
    detector
    interface
    server
    ${BOOST_LIBS}
    ${OPENCV_LIBS}
    ${GLOG_LIBS}
    ${WORKFLOW_LIBS}
)

## add shang tang privacy blur task server
#add_executable(blur_privacy.out tmp_tools/blur_privacy.cpp)
#target_link_libraries(
#    blur_privacy.out
#    common
#    detector
#    interface
#    ${BOOST_LIBS}
#    ${OPENCV_LIBS}
#    ${GLOG_LIBS}
#    ${WORKFLOW_LIBS}
#)

# add pano privacy blur tool
add_executable(blur_pano_privacy.out tmp_tools/blur_pano_privacy.cpp)
target_link_libraries(
    blur_pano_privacy.out
    common
    detector
    interface
    ${BOOST_LIBS}
    ${OPENCV_LIBS}
    ${GLOG_LIBS}
    ${WORKFLOW_LIBS}
)

# add lama mask prepare tool
add_executable(prepare_lama_mask.out tmp_tools/prepare_lama_mask.cpp)
target_link_libraries(
    prepare_lama_mask.out
    common
    detector
    interface
    ${BOOST_LIBS}
    ${OPENCV_LIBS}
    ${GLOG_LIBS}
    ${WORKFLOW_LIBS}
)

# add cmos obstacle mask detection tool
add_executable(cmos_obstacle_mask_detection.out tmp_tools/detect_cmos_obstacle_mask.cpp)
target_link_libraries(
    cmos_obstacle_mask_detection.out
    common
    detector
    interface
    ${BOOST_LIBS}
    ${OPENCV_LIBS}
    ${GLOG_LIBS}
    ${WORKFLOW_LIBS}
)

# add lowlight image enhancement tool
add_executable(enhance_lowlight_image.out tmp_tools/enhance_lowlight_image.cpp)
target_link_libraries(
    enhance_lowlight_image.out
    common
    detector
    interface
    ${BOOST_LIBS}
    ${OPENCV_LIBS}
    ${GLOG_LIBS}
    ${WORKFLOW_LIBS}
)

# add st privacy detection server
add_executable(st_privacy_detection_server.out web_server/st_privacy_detection_server.cpp)
target_link_libraries(
    st_privacy_detection_server.out
    common
    detector
    interface
    server
    ${BOOST_LIBS}
    ${OPENCV_LIBS}
    ${GLOG_LIBS}
    ${WORKFLOW_LIBS}
)
