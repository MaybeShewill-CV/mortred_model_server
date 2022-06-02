# build factory libs

include_directories(
    ${PROJECT_ROOT_DIR}/3rd_party/include
    ${PROJECT_ROOT_DIR}/src
)

file(GLOB_RECURSE FACTORY_LIB_SRC ${CMAKE_CURRENT_LIST_DIR}/*)
add_library(factory SHARED ${FACTORY_LIB_SRC})
target_link_libraries(
    factory
    common
)
set_target_properties(factory PROPERTIES LINKER_LANGUAGE CXX)