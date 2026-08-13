# build factory libs

file(GLOB_RECURSE FACTORY_LIB_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/*)
add_library(factory SHARED ${FACTORY_LIB_SRC})
target_include_directories(
    factory PUBLIC
    ${PROJECT_ROOT_DIR}/src
)
target_include_directories(
    factory SYSTEM PUBLIC
    ${PROJECT_ROOT_DIR}/3rd_party/include
)
target_link_libraries(
    factory
    common
    models
    server
)
set_target_properties(factory PROPERTIES LINKER_LANGUAGE CXX)
