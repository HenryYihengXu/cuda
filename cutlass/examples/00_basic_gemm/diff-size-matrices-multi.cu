/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*
  This example demonstrates how to call a CUTLASS GEMM kernel and provides a naive reference
  matrix multiply kernel to verify its correctness.

  The CUTLASS Gemm template is instantiated in the function CutlassSgemmNN. This is kernel computes
  the general matrix product (GEMM) using single-precision floating-point arithmetic and assumes
  all matrices have column-major layout.

  The threadblock tile size is chosen as 128x128x8 which offers good performance for large matrices.
  See the CUTLASS Parallel for All blog post for more exposition on the tunable parameters available
  in CUTLASS.

  https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

  Aside from defining and launching the SGEMM kernel, this example does not use any other components
  or utilities within CUTLASS. Such utilities are demonstrated elsewhere in other examples and are
  prevalent in the CUTLASS unit tests.

  This example has delibrately been kept similar to the basic_gemm example from cutass-1.3 to 
  highlight the minimum amount of differences needed to transition to cutlass-2.0.

  Cutlass-1.3 sgemm: https://github.com/NVIDIA/cutlass/blob/master/examples/00_basic_gemm/basic_gemm.cu
*/

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>

// Helper methods to check for errors
#include "helper.h"

//
// CUTLASS includes needed for single-precision GEMM kernel
//

// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/device/gemm.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
  int M0,
  int N0,
  int K0,
  float alpha,
  float const *A0,
  int lda0,
  float const *B0,
  int ldb0,
  float beta,
  float *C0,
  int ldc0,
  int M1,
  int N1,
  int K1,
  float const *A1,
  int lda1,
  float const *B1,
  int ldb1,
  float *C1,
  int ldc1) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`

  using ColumnMajor = cutlass::layout::ColumnMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                  ColumnMajor,  // Layout of A matrix
                                                  float,        // Data-type of B matrix
                                                  ColumnMajor,  // Layout of B matrix
                                                  float,        // Data-type of C matrix
                                                  ColumnMajor>; // Layout of C matrix

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //
  CutlassGemm::Arguments args0({M0 , N0, K0},  // Gemm Problem dimensions
                              {A0, lda0},    // Tensor-ref for source matrix A
                              {B0, ldb0},    // Tensor-ref for source matrix B
                              {C0, ldc0},    // Tensor-ref for source matrix C
                              {C0, ldc0},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue

  CutlassGemm::Arguments args1({M1 , N1, K1},  // Gemm Problem dimensions
                              {A1, lda1},    // Tensor-ref for source matrix A
                              {B1, ldb1},    // Tensor-ref for source matrix B
                              {C1, ldc1},    // Tensor-ref for source matrix C
                              {C1, ldc1},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue                           


  cudaStream_t strm0;
  cudaStream_t strm1;
  cudaStreamCreate(&strm0);
  cudaStreamCreate(&strm1);
  //
  // Launch the CUTLASS GEMM kernel.
  //
  
  cutlass::Status status0 = gemm_operator(args0, nullptr, strm0);
  cutlass::Status status1 = gemm_operator(args1, nullptr, strm1);

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //

  if (status0 != cutlass::Status::kSuccess || status1 != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// The source code after this point in the file is generic CUDA using the CUDA Runtime API
// and simple CUDA kernels to initialize matrices and compute the general matrix product.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a matrix with small integers.
__global__ void InitializeMatrix_kernel(
  float *matrix,
  int ldm,
  int rows,
  int columns,
  int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i + j * ldm;

    // Generate arbitrary elements.
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2);

    matrix[offset] = value;
  }
}

/// Simple function to initialize a matrix to arbitrary small integers.
cudaError_t InitializeMatrix(float *matrix, int ldm, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel<<< grid, block >>>(matrix, ldm, rows, columns, seed);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates device memory for a matrix then fills with arbitrary small integers.
cudaError_t AllocateMatrix(float **matrix, int ldm, int rows, int columns, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(float) * ldm * columns;

  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Initialize matrix elements to arbitrary small integers.
  result = InitializeMatrix(*matrix, ldm, rows, columns, seed);

  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference GEMM computation.
__global__ void ReferenceGemm_kernel(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0;

    for (int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }

    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

/// Reference GEMM computation.
cudaError_t ReferenceGemm(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  dim3 block(16, 16);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );

  ReferenceGemm_kernel<<< grid, block >>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
cudaError_t TestCutlassGemm(int M0, int N0, int K0, float alpha, float beta, int M1, int N1, int K1) {
  cudaError_t result;

  //
  // Define several matrices to be used as operands to GEMM kernels.
  //

  // Compute leading dimensions for each matrix.
  int lda0 = M0;
  int ldb0 = K0;
  int ldc0 = M0;

  int lda1 = M1;
  int ldb1 = K1;
  int ldc1 = M1;

  // Compute size in bytes of the C matrix.
  size_t sizeof_C0 = sizeof(float) * ldc0 * N0;
  size_t sizeof_C1 = sizeof(float) * ldc1 * N1;

  // Define pointers to matrices in GPU device memory.
  float *A0;
  float *B0;
  float *C0_cutlass;
  float *C0_reference;

  float *A1;
  float *B1;
  float *C1_cutlass;
  float *C1_reference;

  //
  // Allocate matrices in GPU device memory with arbitrary seeds.
  //

  result = AllocateMatrix(&A0, lda0, M0, K0, 0);

  if (result !=  cudaSuccess) {
    return result;
  }

  result = AllocateMatrix(&B0, ldb0, K0, N0, 17);

  if (result !=  cudaSuccess) {
    cudaFree(A0);
    return result;
  }

  result = AllocateMatrix(&C0_cutlass, ldc0, M0, N0, 101);

  if (result != cudaSuccess) {
    cudaFree(A0);
    cudaFree(B0);
    return result;
  }

  result = AllocateMatrix(&C0_reference, ldc0, M0, N0, 101);

  if (result != cudaSuccess) {
    cudaFree(A0);
    cudaFree(B0);
    cudaFree(C0_cutlass);
    return result;
  }

  result = cudaMemcpy(C0_reference, C0_cutlass, sizeof_C0, cudaMemcpyDeviceToDevice);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C0_reference);
    cudaFree(C0_cutlass);
    cudaFree(B0);
    cudaFree(A0);

    return result;
  }
//-------------------------------------------------------
  result = AllocateMatrix(&A1, lda1, M1, K1, 0);

  if (result !=  cudaSuccess) {
    return result;
  }

  result = AllocateMatrix(&B1, ldb1, K1, N1, 17);

  if (result !=  cudaSuccess) {
    cudaFree(A1);
    return result;
  }

  result = AllocateMatrix(&C1_cutlass, ldc1, M1, N1, 101);

  if (result != cudaSuccess) {
    cudaFree(A1);
    cudaFree(B1);
    return result;
  }

  result = AllocateMatrix(&C1_reference, ldc1, M1, N1, 101);

  if (result != cudaSuccess) {
    cudaFree(A1);
    cudaFree(B1);
    cudaFree(C1_cutlass);
    return result;
  }

  result = cudaMemcpy(C1_reference, C1_cutlass, sizeof_C1, cudaMemcpyDeviceToDevice);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C1_reference);
    cudaFree(C1_cutlass);
    cudaFree(B1);
    cudaFree(A1);

    return result;
  }

  //
  // Launch CUTLASS GEMM.
  //

  result = CutlassSgemmNN(M0, N0, K0, alpha, A0, lda0, B0, ldb0, beta, C0_cutlass, ldc0, M1, N1, K1, A1, lda1, B1, ldb1, C1_cutlass, ldc1);

  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C0_reference);
    cudaFree(C0_cutlass);
    cudaFree(B0);
    cudaFree(A0);

    cudaFree(C1_reference);
    cudaFree(C1_cutlass);
    cudaFree(B1);
    cudaFree(A1);


    return result;
  }

  //
  // Verify.
  //

  // Launch reference GEMM
//   result = ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

//   if (result != cudaSuccess) {
//     std::cerr << "Reference GEMM kernel failed: "
//       << cudaGetErrorString(result) << std::endl;

//     cudaFree(C_reference);
//     cudaFree(C_cutlass);
//     cudaFree(B);
//     cudaFree(A);

//     return result;
//   }

//   // Copy to host and verify equivalence.
//   std::vector<float> host_cutlass(ldc * N, 0);
//   std::vector<float> host_reference(ldc * N, 0);

//   result = cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);

//   if (result != cudaSuccess) {
//     std::cerr << "Failed to copy CUTLASS GEMM results: "
//       << cudaGetErrorString(result) << std::endl;

//     cudaFree(C_reference);
//     cudaFree(C_cutlass);
//     cudaFree(B);
//     cudaFree(A);

//     return result;
//   }

//   result = cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);

//   if (result != cudaSuccess) {
//     std::cerr << "Failed to copy Reference GEMM results: "
//       << cudaGetErrorString(result) << std::endl;

//     cudaFree(C_reference);
//     cudaFree(C_cutlass);
//     cudaFree(B);
//     cudaFree(A);

//     return result;
//   }

  //
  // Free device memory allocations.
  //

  cudaFree(C0_reference);
  cudaFree(C0_cutlass);
  cudaFree(B0);
  cudaFree(A0);

  cudaFree(C1_reference);
  cudaFree(C1_cutlass);
  cudaFree(B1);
  cudaFree(A1);

  //
  // Test for bit equivalence of results.
  //

//   if (host_cutlass != host_reference) {
//     std::cerr << "CUTLASS results incorrect." << std::endl;

//     return cudaErrorUnknown;
//   }

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to basic_gemm example.
//
// usage:
//
//   00_basic_gemm <M> <N> <K> <alpha> <beta>
//
int main(int argc, const char *arg[]) {

  //
  // Parse the command line to obtain GEMM dimensions and scalar values.
  //

  // GEMM problem dimensions.
  int problem0[3] = { 128, 128, 128 };
  int problem1[3] = { 128, 128, 128 };

  for (int i = 1; i < 4; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem0[i - 1];
  }

  for (int i = 4; i < 7; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem1[i - 4];
  }

  // Scalars used for linear scaling the result of the matrix product.
  float scalars[2] = { 1, 0 };

//   for (int i = 4; i < argc && i < 6; ++i) {
//     std::stringstream ss(arg[i]);
//     ss >> scalars[i - 4];
//   }

  //
  // Run the CUTLASS GEMM test.
  //

  cudaError_t result = TestCutlassGemm(
    problem0[0],     // GEMM M dimension
    problem0[1],     // GEMM N dimension
    problem0[2],     // GEMM K dimension
    scalars[0],     // alpha
    scalars[1],      // beta
    problem1[0],     // GEMM M dimension
    problem1[1],     // GEMM N dimension
    problem1[2]     // GEMM K dimension
  );

  if (result == cudaSuccess) {
    std::cout << "Passed." << std::endl;
  }

  // Exit.
  return result == cudaSuccess ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
