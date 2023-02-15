# build server libs

include_directories(
    ${PROJECT_ROOT_DIR}/3rd_party/include
    ${PROJECT_ROOT_DIR}/src
)

file(GLOB_RECURSE SERVER_LIB_SRC ${CMAKE_CURRENT_LIST_DIR}/*)
add_library(registration SHARED ${SERVER_LIB_SRC})
target_link_libraries(
    registration
    common
    ${GLOG_LIBRARIES}
    ${FMT_LIBRARIES}
)
set_target_properties(registration PROPERTIES LINKER_LANGUAGE CXX)