# build apps

# benchmark / server app assembly is identical; share one function (previously two verbatim copies)
function(add_mortred_app src)
    get_filename_component(file_name ${src} NAME_WE)
    add_executable(${file_name}.out ${src})
    target_link_libraries(${file_name}.out common models factory vendored::workflow glog::glog)
    # Apps instantiate .inl templates (with old OpenCV API calls); deprecation suppression goes per-target
    target_compile_options(${file_name}.out PRIVATE -Wno-deprecated-declarations)
endfunction()

# --- benchmark apps ---
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/classification/densenet_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/classification/dinov2_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/classification/mobilenetv2_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/classification/resnet_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/clip/openai_clip_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/diffusion/cls_cond_ddim_sampler_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/diffusion/ddim_sampler_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/diffusion/ddpm_sampler_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/diffusion/ldm_sampler_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/enhancement/attentivegan_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/enhancement/enlightengan_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/enhancement/real_esrgan_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/feature_point/lightglue_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/feature_point/superpoint_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/matting/modnet_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/matting/ppmatting_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/mono_depth_estimation/depth_anything_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/mono_depth_estimation/metric3d_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/mot/bytetrack_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/object_detection/centerface_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/object_detection/libface_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/object_detection/nanodet_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/object_detection/yolov5_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/object_detection/yolov6_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/object_detection/yolov7_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/object_detection/yolov8_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/ocr/dbnet_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/sam/fast_sam_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/sam/sam_amg_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/sam/sam_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/segmentation/bisenetv2_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/segmentation/hrnet_segmentation_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/segmentation/msocrnet_benchmark.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/model_benchmark/segmentation/pphumanseg_benchmark.cpp)

# --- server apps ---
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/classification/densenet_classification_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/classification/mobilenetv2_classification_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/classification/resnet_classification_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/enhancement/attentive_gan_derain_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/enhancement/enlighten_gan_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/enhancement/real_esrgan_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/feature_point/superpoint_fp_det_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/matting/modnet_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/matting/pp_matting_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/mono_depth_estimation/depth_anything_estimation_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/mono_depth_estimation/metric3d_estimation_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/object_detection/centerface_detection_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/object_detection/libface_detection_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/object_detection/nanodet_detection_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/object_detection/yolov5_detection_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/object_detection/yolov6_detection_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/object_detection/yolov7_detection_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/object_detection/yolov8_detection_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/ocr/dbnet_text_region_detect_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/scene_segmentation/bisenetv2_segmentation_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/scene_segmentation/hrnet_segmentation_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/scene_segmentation/pphuman_segmentation_server.cpp)

# --- diffusion / SAM AMG servers (async long-task, P0-8) ---
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/diffusion/ddpm_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/diffusion/ddim_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/diffusion/cls_cond_ddim_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/diffusion/ldm_server.cpp)
add_mortred_app(${CMAKE_CURRENT_LIST_DIR}/server/segment_anything/sam_amg_server.cpp)

# --- control-plane executables ---
# src/apps produces every executable; the implementations live in src/control
# (thin mains here + *_app.cpp compiled in, mirroring the models/.inl pattern).
if(MORTRED_WORKFLOW_AVAILABLE)
    add_executable(mortred-gateway.out
        ${CMAKE_CURRENT_LIST_DIR}/control/gateway.cpp
        ${PROJECT_ROOT_DIR}/src/control/gateway/gateway_app.cpp)
    target_include_directories(mortred-gateway.out PRIVATE ${PROJECT_ROOT_DIR}/src)
    target_include_directories(mortred-gateway.out SYSTEM PRIVATE ${PROJECT_ROOT_DIR}/3rd_party/include)
    target_link_libraries(mortred-gateway.out PRIVATE control vendored::workflow vendored::crypto Threads::Threads)

    add_executable(mortred-supervisor.out
        ${CMAKE_CURRENT_LIST_DIR}/control/supervisor.cpp
        ${PROJECT_ROOT_DIR}/src/control/supervisor/supervisor_app.cpp)
    target_include_directories(mortred-supervisor.out PRIVATE ${PROJECT_ROOT_DIR}/src)
    target_include_directories(mortred-supervisor.out SYSTEM PRIVATE ${PROJECT_ROOT_DIR}/3rd_party/include)
    target_link_libraries(mortred-supervisor.out PRIVATE control common vendored::workflow Threads::Threads)

    add_executable(mortredctl.out
        ${CMAKE_CURRENT_LIST_DIR}/control/mortredctl.cpp
        ${PROJECT_ROOT_DIR}/src/control/cli/cli_app.cpp)
    target_include_directories(mortredctl.out PRIVATE ${PROJECT_ROOT_DIR}/src)
    target_include_directories(mortredctl.out SYSTEM PRIVATE ${PROJECT_ROOT_DIR}/3rd_party/include)
    target_link_libraries(mortredctl.out PRIVATE control common vendored::workflow Threads::Threads)

    foreach(control_bin mortred-gateway.out mortred-supervisor.out mortredctl.out)
        set_target_properties(${control_bin} PROPERTIES
            BUILD_RPATH "${PROJECT_ROOT_DIR}/3rd_party/libs;${PROJECT_ROOT_DIR}/_lib")
    endforeach()
endif()

