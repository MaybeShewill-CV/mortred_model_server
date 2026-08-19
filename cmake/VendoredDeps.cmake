# VendoredDeps.cmake —— vendored 二进制依赖的显式声明（替代 file(GLOB)）
#
# 设计目标：
# 1. 符号链接链折叠为单一精确文件（libworkflow.so/.so.0/.so.0.10.9 → 一个 target）
# 2. 缺失即 configure 期致命错并给出修复命令，而不是链接/运行期的晦涩失败
# 3. 精确 soname 钉版本（与 scripts/install_deps.sh 版本矩阵对齐），
#    错误版本（如 TRT 10）混入会被"找不到 libnvinfer.so.8"直接拦截
# 4. 精确文件变更自动触发重新配置（CONFIGURE_DEPENDS），无 GLOB 陈旧性

set(MORTRED_3RD_LIBS "${PROJECT_ROOT_DIR}/3rd_party/libs" CACHE PATH
    "Directory holding the vendored shared libraries installed by scripts/install_deps.sh")

# 运行时组件（providers/builder_resource/cudnn 等）由动态加载器按需 dlopen，
# 不在此声明为链接项。

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

# 可选探测（tests-only 场景）：vendored workflow 存在则契约 e2e 测试可运行，
# 不存在（CI vcpkg 路径）则自动跳过——与旧的 if(WORKFLOW_LIBS) 行为一致。
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
