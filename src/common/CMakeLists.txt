# build common libs

set(COMMON_LIB_SRC
    ${CMAKE_CURRENT_LIST_DIR}/time_stamp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/auth_token.h
    ${CMAKE_CURRENT_LIST_DIR}/listen_policy.h
    ${CMAKE_CURRENT_LIST_DIR}/base64.h
    ${CMAKE_CURRENT_LIST_DIR}/cv_utils.h
    ${CMAKE_CURRENT_LIST_DIR}/file_path_util.h
    ${CMAKE_CURRENT_LIST_DIR}/request_envelope.h
    ${CMAKE_CURRENT_LIST_DIR}/response_envelope.h
    ${CMAKE_CURRENT_LIST_DIR}/request_size_limit.h
    ${CMAKE_CURRENT_LIST_DIR}/status_code.h
    ${CMAKE_CURRENT_LIST_DIR}/time_stamp.h
)
add_library(common SHARED ${COMMON_LIB_SRC})
target_link_libraries(
    common
    PUBLIC mortred_common_includes
    Threads::Threads
    glog::glog
)
set_target_properties(common PROPERTIES LINKER_LANGUAGE CXX)
