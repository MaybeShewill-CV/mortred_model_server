# build factory libs

# Every row of every catalog lives in a *_task.h; the library compiles no
# sources, so the header list is globbed (CONFIGURE_DEPENDS re-globs when a new
# family header appears instead of requiring a manual edit).
file(GLOB FACTORY_HEADERS CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/*.h)
add_library(factory SHARED ${FACTORY_HEADERS})
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
