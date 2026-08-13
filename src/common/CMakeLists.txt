# build common libs

file(GLOB_RECURSE COMMON_LIB_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/*)
add_library(common SHARED ${COMMON_LIB_SRC})
target_link_libraries(
    common
    PUBLIC mortred_common_includes
    Threads::Threads
    ${OPENCV_LIBS}
)
set_target_properties(common PROPERTIES LINKER_LANGUAGE CXX)
