# build server libs

set(SERVER_LIB_SRC
    ${CMAKE_CURRENT_LIST_DIR}/abstract_server.h
    ${CMAKE_CURRENT_LIST_DIR}/base_server_impl.h
    ${CMAKE_CURRENT_LIST_DIR}/generic_cv_server.h
    ${CMAKE_CURRENT_LIST_DIR}/http_status.h
    ${CMAKE_CURRENT_LIST_DIR}/output_options.h
    ${CMAKE_CURRENT_LIST_DIR}/prometheus_metrics.h
    ${CMAKE_CURRENT_LIST_DIR}/rate_limiter.h
    ${CMAKE_CURRENT_LIST_DIR}/request_envelope.h
    ${CMAKE_CURRENT_LIST_DIR}/response_serializers.h
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
    vendored::workflow
    glog::glog
)
set_target_properties(server PROPERTIES LINKER_LANGUAGE CXX)
