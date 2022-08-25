# build common libs

include_directories(
    ${PROJECT_ROOT_DIR}/3rd_party/include
)

file(GLOB_RECURSE COMMON_LIB_SRC ${CMAKE_CURRENT_LIST_DIR}/*)
add_library(common SHARED ${COMMON_LIB_SRC})
target_link_libraries(
    common
    Threads::Threads
    ${OPENCV_LIBS}
    ${BASE64_LIBRARIES}
)
set_target_properties(common PROPERTIES LINKER_LANGUAGE CXX)
