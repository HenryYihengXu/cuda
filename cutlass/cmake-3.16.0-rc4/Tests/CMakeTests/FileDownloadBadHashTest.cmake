if(NOT "/users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/Tests/CMakeTests" MATCHES "^/")
  set(slash /)
endif()
set(url "file://${slash}/users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/Tests/CMakeTests/FileDownloadInput.png")
set(dir "/users/HenryXu/cuda/cutlass/cmake-3.16.0-rc4/Tests/CMakeTests/downloads")

file(DOWNLOAD
  ${url}
  ${dir}/file3.png
  TIMEOUT 2
  STATUS status
  EXPECTED_HASH SHA1=5555555555555555555555555555555555555555
  )
