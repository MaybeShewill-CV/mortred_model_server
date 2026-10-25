# Default `check` driver.
# Excluded names live in test/CHECK_SSOT.md (scripts/check_check_ssot.py).
#
# cmake -DCTEST=<ctest> -DTEST_DIR=<test binary dir> -DJUNIT=<path-or-empty>
#       -P test/run_default_check.cmake
#
# MORTRED_CHECK_HEAVY=1 runs everything except label valgrind.
# Default runs unit + contract only: heavy-golden, e2e and stress stay on their
# own lanes (MORTRED_CHECK_HEAVY=1, ctest -L e2e, ctest -L stress).

if(NOT DEFINED CTEST OR CTEST STREQUAL "")
    message(FATAL_ERROR "run_default_check: -DCTEST= is required")
endif()
if(NOT DEFINED TEST_DIR OR TEST_DIR STREQUAL "")
    message(FATAL_ERROR "run_default_check: -DTEST_DIR= is required")
endif()
if(NOT EXISTS "${TEST_DIR}/CTestTestfile.cmake")
    message(FATAL_ERROR "run_default_check: no CTestTestfile.cmake in ${TEST_DIR}")
endif()

set(_heavy "$ENV{MORTRED_CHECK_HEAVY}")
if(_heavy STREQUAL "1")
    set(_exclude "valgrind")
    message(STATUS "MORTRED_CHECK_HEAVY=1: check runs heavy-golden, e2e and stress, still excludes valgrind")
else()
    set(_exclude "valgrind|heavy-golden|e2e|stress")
    message(STATUS "check excludes labels valgrind, heavy-golden, e2e and stress (MORTRED_CHECK_HEAVY=1 for the first three, ctest -L e2e / -L stress for the last two)")
endif()

set(_cmd "${CTEST}" --output-on-failure -LE "${_exclude}")
if(DEFINED JUNIT AND NOT JUNIT STREQUAL "")
    list(APPEND _cmd --output-junit "${JUNIT}")
endif()

execute_process(
    COMMAND ${_cmd}
    WORKING_DIRECTORY "${TEST_DIR}"
    RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "check: ctest -LE ${_exclude} exited ${_rc}")
endif()
