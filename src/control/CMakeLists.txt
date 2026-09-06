# mortred control-plane module (mirrors src/models): all implementation lives
# here - the workflow-free supervision core (the `control` shared library,
# unit-tested in tests-only CI) plus the workflow-bound app objects (gateway /
# supervisor *_app.cpp, in the `control_workflow` static library below).
#
# The shared library is deliberately workflow/opencv-free so it builds in every
# configuration; `control_workflow` exists ONLY when the vendored workflow is
# available, and is linked by the src/apps executables and the in-process
# control-plane tests. UI assets are a runtime resource of the supervisor,
# installed to share/mortred/ui by the root install rules.
add_library(control SHARED
    catalog.cpp
    control_config.cpp
    supervisor.cpp
)
target_include_directories(control PUBLIC ${PROJECT_ROOT_DIR}/src)
target_link_libraries(control PUBLIC Threads::Threads)
set_target_properties(control PROPERTIES LINKER_LANGUAGE CXX)

if(MORTRED_WORKFLOW_AVAILABLE)
    add_library(control_workflow STATIC
        gateway/gateway_app.cpp
        supervisor/supervisor_app.cpp
    )
    # `common` propagates the src/ + 3rd_party/include scopes (header-only
    # jinq::common helpers, server/prometheus_metrics.h); vendored::crypto is
    # the inlined OpenSSL SHA-256 of ApiKeyManager
    target_link_libraries(control_workflow
        PUBLIC control common vendored::workflow vendored::crypto Threads::Threads)
    set_target_properties(control_workflow PROPERTIES LINKER_LANGUAGE CXX)
endif()
