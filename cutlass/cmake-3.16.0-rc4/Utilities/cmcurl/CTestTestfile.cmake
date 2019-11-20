# CMake generated Testfile for 
# Source directory: /users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/Utilities/cmcurl
# Build directory: /users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/Utilities/cmcurl
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(curl "curltest" "http://open.cdash.org/user.php")
set_tests_properties(curl PROPERTIES  _BACKTRACE_TRIPLES "/users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/Utilities/cmcurl/CMakeLists.txt;1300;add_test;/users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/Utilities/cmcurl/CMakeLists.txt;0;")
subdirs("lib")
