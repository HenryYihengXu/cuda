# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /users/adarsh/cuda/cutlass

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /users/adarsh/cuda/cutlass/build

# Include any dependencies generated for this target.
include examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir/depend.make

# Include the progress variables for this target.
include examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir/progress.make

# Include the compile flags for this target's objects.
include examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir/flags.make

examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir/05_wmma_gemm_generated_wmma_gemm.cu.o: examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir/05_wmma_gemm_generated_wmma_gemm.cu.o.depend
examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir/05_wmma_gemm_generated_wmma_gemm.cu.o: examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir/05_wmma_gemm_generated_wmma_gemm.cu.o.cmake
examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir/05_wmma_gemm_generated_wmma_gemm.cu.o: ../examples/05_wmma_gemm/wmma_gemm.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/users/adarsh/cuda/cutlass/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir/05_wmma_gemm_generated_wmma_gemm.cu.o"
	cd /users/adarsh/cuda/cutlass/build/examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir && /usr/bin/cmake -E make_directory /users/adarsh/cuda/cutlass/build/examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir//.
	cd /users/adarsh/cuda/cutlass/build/examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Release -D generated_file:STRING=/users/adarsh/cuda/cutlass/build/examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir//./05_wmma_gemm_generated_wmma_gemm.cu.o -D generated_cubin_file:STRING=/users/adarsh/cuda/cutlass/build/examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir//./05_wmma_gemm_generated_wmma_gemm.cu.o.cubin.txt -P /users/adarsh/cuda/cutlass/build/examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir//05_wmma_gemm_generated_wmma_gemm.cu.o.cmake

# Object files for target 05_wmma_gemm
05_wmma_gemm_OBJECTS =

# External object files for target 05_wmma_gemm
05_wmma_gemm_EXTERNAL_OBJECTS = \
"/users/adarsh/cuda/cutlass/build/examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir/05_wmma_gemm_generated_wmma_gemm.cu.o"

examples/05_wmma_gemm/05_wmma_gemm: examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir/05_wmma_gemm_generated_wmma_gemm.cu.o
examples/05_wmma_gemm/05_wmma_gemm: examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir/build.make
examples/05_wmma_gemm/05_wmma_gemm: /usr/local/cuda-9.0/lib64/libcudart_static.a
examples/05_wmma_gemm/05_wmma_gemm: /usr/lib/x86_64-linux-gnu/librt.so
examples/05_wmma_gemm/05_wmma_gemm: examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/users/adarsh/cuda/cutlass/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable 05_wmma_gemm"
	cd /users/adarsh/cuda/cutlass/build/examples/05_wmma_gemm && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/05_wmma_gemm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir/build: examples/05_wmma_gemm/05_wmma_gemm

.PHONY : examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir/build

examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir/requires:

.PHONY : examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir/requires

examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir/clean:
	cd /users/adarsh/cuda/cutlass/build/examples/05_wmma_gemm && $(CMAKE_COMMAND) -P CMakeFiles/05_wmma_gemm.dir/cmake_clean.cmake
.PHONY : examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir/clean

examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir/depend: examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir/05_wmma_gemm_generated_wmma_gemm.cu.o
	cd /users/adarsh/cuda/cutlass/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /users/adarsh/cuda/cutlass /users/adarsh/cuda/cutlass/examples/05_wmma_gemm /users/adarsh/cuda/cutlass/build /users/adarsh/cuda/cutlass/build/examples/05_wmma_gemm /users/adarsh/cuda/cutlass/build/examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/05_wmma_gemm/CMakeFiles/05_wmma_gemm.dir/depend

