# CMake generated Testfile for 
# Source directory: /users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4
# Build directory: /users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
include("/users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/Tests/EnforceConfig.cmake")
add_test(SystemInformationNew "/users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/bin/cmake" "--system-information" "-G" "Unix Makefiles")
set_tests_properties(SystemInformationNew PROPERTIES  _BACKTRACE_TRIPLES "/users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/CMakeLists.txt;825;add_test;/users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/CMakeLists.txt;0;")
subdirs("Source/kwsys")
subdirs("Utilities/std")
subdirs("Utilities/KWIML")
subdirs("Utilities/cmlibrhash")
subdirs("Utilities/cmzlib")
subdirs("Utilities/cmcurl")
subdirs("Utilities/cmexpat")
subdirs("Utilities/cmbzip2")
subdirs("Utilities/cmzstd")
subdirs("Utilities/cmliblzma")
subdirs("Utilities/cmlibarchive")
subdirs("Utilities/cmjsoncpp")
subdirs("Utilities/cmlibuv")
subdirs("Source/CursesDialog/form")
subdirs("Source")
subdirs("Utilities")
subdirs("Tests")
subdirs("Auxiliary")
