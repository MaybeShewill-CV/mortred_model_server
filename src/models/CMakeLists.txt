# build models libs

include_directories(
    ${PROJECT_ROOT_DIR}/3rd_party/include
    ${PROJECT_ROOT_DIR}/src
)

file(GLOB MNN_LIBS ${PROJECT_ROOT_DIR}/3rd_party/libs/*MNN*so*)
file(GLOB_RECURSE OPENCV_LIBS ${PROJECT_ROOT_DIR}/3rd_party/libs/*opencv*so)
file(GLOB_RECURSE DETECTOR_LIB_SRC ${CMAKE_CURRENT_LIST_DIR}/*)
add_library(models SHARED ${DETECTOR_LIB_SRC})
target_link_libraries(
    models
    common
    ${MNN_LIBS}
    ${OPENCV_LIBS}
)
set_target_properties(models PROPERTIES LINKER_LANGUAGE CXX)
