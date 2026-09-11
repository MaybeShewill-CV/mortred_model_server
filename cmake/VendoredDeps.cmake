# VendoredDeps.cmake - explicit declarations of vendored binary deps (replaces file(GLOB))
#
# Design goals:
# 1. Symlink chains collapse to a single exact file (libworkflow.so/.so.0/.so.0.10.9 -> one target)
# 2. Missing libs fail fatally at configure time with a fix command, not obscure link/run errors
# 3. Exact sonames pin versions (aligned with scripts/install_deps.sh: CUDA 12 /
#    TensorRT 10.3 / ORT 1.29). leftover TRT 8 / ORT 1.18 files fail configure.
# 4. Exact-file changes trigger automatic reconfigure (CONFIGURE_DEPENDS), no GLOB staleness

set(MORTRED_3RD_LIBS "${PROJECT_ROOT_DIR}/3rd_party/libs" CACHE PATH
    "Directory holding the vendored shared libraries installed by scripts/install_deps.sh")

# Runtime components (providers/builder_resource/cudnn etc.) are dlopened on demand
# by the dynamic loader; not declared as link items here.

function(mortred_import_shared name soname)
    find_library(mortred_lib_${name} NAMES ${soname}
                 HINTS "${MORTRED_3RD_LIBS}" NO_DEFAULT_PATH)
    if(NOT mortred_lib_${name})
        message(FATAL_ERROR
            "vendored lib '${soname}' not found in ${MORTRED_3RD_LIBS}\n"
            "fix: ./scripts/install_deps.sh --all   (then re-run cmake configure)")
    endif()
    add_library(vendored::${name} SHARED IMPORTED GLOBAL)
    set_target_properties(vendored::${name} PROPERTIES
        IMPORTED_LOCATION "${mortred_lib_${name}}")
    set_property(DIRECTORY APPEND PROPERTY
        CMAKE_CONFIGURE_DEPENDS "${mortred_lib_${name}}")
    message(STATUS "vendored ${name}: ${mortred_lib_${name}}")
endfunction()

# libcrypto.so.3 on ubuntu 22.04 / GPU Docker; .so.1.1 on older trees.
function(mortred_import_crypto)
    find_library(mortred_lib_crypto3 NAMES libcrypto.so.3
                 HINTS "${MORTRED_3RD_LIBS}" NO_DEFAULT_PATH)
    find_library(mortred_lib_crypto11 NAMES libcrypto.so.1.1
                 HINTS "${MORTRED_3RD_LIBS}" NO_DEFAULT_PATH)
    if(mortred_lib_crypto3)
        set(_mortred_crypto "${mortred_lib_crypto3}")
    elseif(mortred_lib_crypto11)
        set(_mortred_crypto "${mortred_lib_crypto11}")
    else()
        message(FATAL_ERROR
            "vendored libcrypto.so.3 or libcrypto.so.1.1 not found in ${MORTRED_3RD_LIBS}\n"
            "fix: ./scripts/install_deps.sh --all   (then re-run cmake configure)")
    endif()
    add_library(vendored::crypto SHARED IMPORTED GLOBAL)
    set_target_properties(vendored::crypto PROPERTIES
        IMPORTED_LOCATION "${_mortred_crypto}")
    set_property(DIRECTORY APPEND PROPERTY
        CMAKE_CONFIGURE_DEPENDS "${_mortred_crypto}")
    message(STATUS "vendored crypto: ${_mortred_crypto}")
endfunction()

# Optional probe (tests-only case): vendored workflow present => e2e contract tests can run;
# absent (CI vcpkg path) => skipped automatically, matching the old if(WORKFLOW_LIBS) behavior.
macro(mortred_probe_workflow)
    find_library(mortred_workflow_lib NAMES libworkflow.so
                 HINTS "${MORTRED_3RD_LIBS}" NO_DEFAULT_PATH)
    if(mortred_workflow_lib)
        add_library(vendored::workflow SHARED IMPORTED GLOBAL)
        set_target_properties(vendored::workflow PROPERTIES
            IMPORTED_LOCATION "${mortred_workflow_lib}")
        set_property(DIRECTORY APPEND PROPERTY
            CMAKE_CONFIGURE_DEPENDS "${mortred_workflow_lib}")
        set(MORTRED_WORKFLOW_AVAILABLE TRUE)
        message(STATUS "vendored workflow: ${mortred_workflow_lib}")
    else()
        set(MORTRED_WORKFLOW_AVAILABLE FALSE)
        message(STATUS "vendored workflow: not found (e2e contract test will be skipped)")
    endif()
endmacro()
