# Execute each test listed in:
#
set(scriptname "/users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/Tests/CMakeTests/MathTestScript.cmake")
set(number_of_tests_expected 4)

include("/users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/Tests/CMakeTests/ExecuteScriptTests.cmake")
execute_all_script_tests(${scriptname} number_of_tests_executed)

# And verify that number_of_tests_executed is at least as many as we know
# about as of this writing...
#
message(STATUS "scriptname='${scriptname}'")
message(STATUS "number_of_tests_executed='${number_of_tests_executed}'")
message(STATUS "number_of_tests_expected='${number_of_tests_expected}'")

if(number_of_tests_executed LESS number_of_tests_expected)
  message(FATAL_ERROR "error: some test cases were skipped")
endif()