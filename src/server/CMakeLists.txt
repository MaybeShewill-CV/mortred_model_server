# build server libs

include_directories(
    ${PROJECT_ROOT_DIR}/3rd_party/include
    ${PROJECT_ROOT_DIR}/src
    ${PROJECT_ROOT_DIR}/src/server
)

file(GLOB_RECURSE SERVER_LIB_SRC ${CMAKE_CURRENT_LIST_DIR}/*)
file(GLOB_RECURSE WORKFLOW_LIBS ${PROJECT_ROOT_DIR}/3rd_party/libs/*workflow*so)
add_library(server SHARED ${SERVER_LIB_SRC})
target_link_libraries(
    server
    common
    models
    ${WORKFLOW_LIBS}
)
set_target_properties(server PROPERTIES LINKER_LANGUAGE CXX)