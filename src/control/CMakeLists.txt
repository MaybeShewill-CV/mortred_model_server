# mortred control-plane module (mirrors src/models): all implementation lives
# here - the workflow-free supervision core (this shared library, unit-tested
# in tests-only CI) plus the workflow-dependent app logic (gateway/supervisor/
# cli *_app.cpp, compiled into the executables built under src/apps, the same
# way models' .inl templates are compiled into their app targets).
#
# The shared library is deliberately workflow/opencv-free so it builds in every
# configuration. UI assets are a runtime resource of the supervisor, installed
# to share/mortred/ui by the root install rules.
add_library(control SHARED
    catalog.cpp
    control_config.cpp
    supervisor.cpp
)
target_include_directories(control PUBLIC ${PROJECT_ROOT_DIR}/src)
target_link_libraries(control PUBLIC Threads::Threads)
set_target_properties(control PROPERTIES LINKER_LANGUAGE CXX)
