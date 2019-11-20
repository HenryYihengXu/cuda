# CMake generated Testfile for 
# Source directory: /users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/Tests/FindThreads
# Build directory: /users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/Tests/FindThreads
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(FindThreads.C-only "/users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/bin/ctest" "--build-and-test" "/users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/Tests/FindThreads/C-only" "/users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/Tests/FindThreads/C-only" "--build-generator" "Unix Makefiles" "--build-makeprogram" "/usr/bin/make" "--build-project" "FindThreads_C-only" "--build-options" "--test-command" "/users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/bin/ctest" "-V")
set_tests_properties(FindThreads.C-only PROPERTIES  _BACKTRACE_TRIPLES "/users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/Tests/FindThreads/CMakeLists.txt;2;add_test;/users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/Tests/FindThreads/CMakeLists.txt;0;")
add_test(FindThreads.CXX-only "/users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/bin/ctest" "--build-and-test" "/users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/Tests/FindThreads/CXX-only" "/users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/Tests/FindThreads/CXX-only" "--build-generator" "Unix Makefiles" "--build-makeprogram" "/usr/bin/make" "--build-project" "FindThreads_CXX-only" "--build-options" "--test-command" "/users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/bin/ctest" "-V")
set_tests_properties(FindThreads.CXX-only PROPERTIES  _BACKTRACE_TRIPLES "/users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/Tests/FindThreads/CMakeLists.txt;2;add_test;/users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/Tests/FindThreads/CMakeLists.txt;0;")
