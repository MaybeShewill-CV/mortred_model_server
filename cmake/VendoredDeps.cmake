# VendoredDeps.cmake - explicit declarations of vendored binary deps (replaces file(GLOB))
#
# Design goals:
# 1. Symlink chains collapse to a single exact file (libworkflow.so/.so.0/.so.0.10.9 -> one target)
# 2. Missing libs fail fatally at configure time with a fix command, not obscure link/run errors
# 3. Exact sonames pin versions (aligned with scripts/install_deps.sh: CUDA 12 /
#    TensorRT 10.3 / ORT 1.29). leftover TRT 8 / ORT 1.18 libs and mismatched
#    ORT headers (ORT_API_VERSION) fail configure (SME-10).
# 4. Exact-file changes trigger automatic reconfigure (CONFIGURE_DEPENDS), no GLOB staleness

set(MORTRED_3RD_LIBS "${PROJECT_ROOT_DIR}/3rd_party/libs" CACHE PATH
    "Directory holding the vendored shared libraries installed by scripts/install_deps.sh")
set(MORTRED_3RD_INCLUDE "${PROJECT_ROOT_DIR}/3rd_party/include" CACHE PATH
    "Directory holding vendored headers installed by scripts/install_deps.sh")

# Runtime components (providers/builder_resource/cudnn etc.) are dlopened on demand
# by the dynamic loader; not declared as link items here.

# Profile-aware fix line (call after MORTRED_BUILD_PROFILE is set).
function(mortred_deps_fix_cmd outvar)
    if(DEFINED MORTRED_BUILD_PROFILE AND MORTRED_BUILD_PROFILE STREQUAL "cpu")
        set(${outvar} "./scripts/install_deps.sh --cpu --all" PARENT_SCOPE)
    else()
        set(${outvar} "./scripts/install_deps.sh --all" PARENT_SCOPE)
    endif()
endfunction()

function(mortred_deps_check_cmd outvar)
    if(DEFINED MORTRED_BUILD_PROFILE AND MORTRED_BUILD_PROFILE STREQUAL "cpu")
        set(${outvar} "./scripts/install_deps.sh --cpu --check" PARENT_SCOPE)
    else()
        set(${outvar} "./scripts/install_deps.sh --check" PARENT_SCOPE)
    endif()
endfunction()

function(mortred_import_shared name soname)
    find_library(mortred_lib_${name} NAMES ${soname}
                 HINTS "${MORTRED_3RD_LIBS}" NO_DEFAULT_PATH)
    if(NOT mortred_lib_${name})
        mortred_deps_fix_cmd(_mortred_fix)
        mortred_deps_check_cmd(_mortred_check)
        message(FATAL_ERROR
            "vendored lib '${soname}' not found in ${MORTRED_3RD_LIBS}\n"
            "fix: ${_mortred_fix}   (then re-run cmake configure)\n"
            "verify: ${_mortred_check}")
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
        mortred_deps_fix_cmd(_mortred_fix)
        # --workflow also installs system runtime libs (libcrypto); keep hint broad.
        mortred_deps_check_cmd(_mortred_check)
        message(FATAL_ERROR
            "vendored libcrypto.so.3 or libcrypto.so.1.1 not found in ${MORTRED_3RD_LIBS}\n"
            "fix: ${_mortred_fix}   (or ./scripts/install_deps.sh --workflow)\n"
            "verify: ${_mortred_check}")
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

# Full build hard-requires workflow (+ crypto imported when workflow is present).
function(mortred_require_workflow_for_full)
    if(MORTRED_WORKFLOW_AVAILABLE)
        return()
    endif()
    mortred_deps_fix_cmd(_mortred_fix)
    mortred_deps_check_cmd(_mortred_check)
    message(FATAL_ERROR
        "full build requires vendored libworkflow.so in ${MORTRED_3RD_LIBS}\n"
        "fix: ${_mortred_fix}   (then re-run cmake configure)\n"
        "verify: ${_mortred_check}")
endfunction()

# ORT tarball 1.N.x ships ORT_API_VERSION == N (e.g. 1.29.0 -> 29).
function(mortred_assert_onnxruntime_pin)
    # leftover 1.18 libs next to the 1.29 soname are a known footgun.
    file(GLOB _mortred_leftover_ort118 "${MORTRED_3RD_LIBS}/libonnxruntime.so.1.18*")
    if(_mortred_leftover_ort118)
        mortred_deps_fix_cmd(_mortred_fix)
        mortred_deps_check_cmd(_mortred_check)
        message(FATAL_ERROR
            "leftover ONNX Runtime 1.18 libraries in ${MORTRED_3RD_LIBS}:\n  ${_mortred_leftover_ort118}\n"
            "fix: ${_mortred_fix}   (or ./scripts/install_deps.sh --onnxruntime)\n"
            "verify: ${_mortred_check}")
    endif()

    set(_mortred_ort_hdr "")
    foreach(_cand
            "${MORTRED_3RD_INCLUDE}/onnxruntime/onnxruntime_c_api.h"
            "${MORTRED_3RD_INCLUDE}/onnxruntime/core/session/onnxruntime_c_api.h")
        if(EXISTS "${_cand}")
            set(_mortred_ort_hdr "${_cand}")
            break()
        endif()
    endforeach()
    if(NOT _mortred_ort_hdr)
        mortred_deps_fix_cmd(_mortred_fix)
        mortred_deps_check_cmd(_mortred_check)
        message(FATAL_ERROR
            "onnxruntime headers missing under ${MORTRED_3RD_INCLUDE}/onnxruntime/ "
            "(need onnxruntime_c_api.h with ORT_API_VERSION)\n"
            "fix: ${_mortred_fix}   (or ./scripts/install_deps.sh --onnxruntime)\n"
            "verify: ${_mortred_check}")
    endif()

    file(STRINGS "${_mortred_ort_hdr}" _mortred_ort_api_lines
         REGEX "^[ \t]*#define[ \t]+ORT_API_VERSION[ \t]+[0-9]+")
    set(_mortred_ort_api "")
    if(_mortred_ort_api_lines)
        list(GET _mortred_ort_api_lines 0 _mortred_ort_api_line)
        string(REGEX REPLACE "^.*ORT_API_VERSION[ \t]+([0-9]+).*$" "\\1"
               _mortred_ort_api "${_mortred_ort_api_line}")
    endif()
    # Pin matches scripts/install_deps.sh ONNXRUNTIME_VER=1.29.0 -> API 29
    # and mortred_import_shared(... libonnxruntime.so.1.29.0).
    set(_mortred_ort_want 29)
    if(NOT _mortred_ort_api STREQUAL "${_mortred_ort_want}")
        mortred_deps_fix_cmd(_mortred_fix)
        mortred_deps_check_cmd(_mortred_check)
        message(FATAL_ERROR
            "onnxruntime headers ORT_API_VERSION=${_mortred_ort_api} "
            "(want ${_mortred_ort_want} for libonnxruntime.so.1.29.0) in ${_mortred_ort_hdr}\n"
            "fix: ${_mortred_fix}   (or ./scripts/install_deps.sh --onnxruntime)\n"
            "verify: ${_mortred_check}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_mortred_ort_hdr}")
    message(STATUS "vendored onnxruntime headers: ORT_API_VERSION=${_mortred_ort_api} (${_mortred_ort_hdr})")
endfunction()
