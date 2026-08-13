# build server libs

file(GLOB_RECURSE SERVER_LIB_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/*)
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
