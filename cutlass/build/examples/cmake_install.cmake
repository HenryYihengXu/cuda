# Install script for directory: /users/adarsh/cuda/cutlass/examples

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/users/adarsh/cuda/cutlass/build/examples/00_basic_gemm/cmake_install.cmake")
  include("/users/adarsh/cuda/cutlass/build/examples/01_tensor_view/cmake_install.cmake")
  include("/users/adarsh/cuda/cutlass/build/examples/02_cutlass_utilities/cmake_install.cmake")
  include("/users/adarsh/cuda/cutlass/build/examples/03_strided_batched_gemm/cmake_install.cmake")
  include("/users/adarsh/cuda/cutlass/build/examples/04_tile_iterator/cmake_install.cmake")
  include("/users/adarsh/cuda/cutlass/build/examples/05_wmma_gemm/cmake_install.cmake")
  include("/users/adarsh/cuda/cutlass/build/examples/06_splitK_gemm/cmake_install.cmake")

endif()

