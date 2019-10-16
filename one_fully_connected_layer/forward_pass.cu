#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t vectorMatrixMulWithCuda(double *c, const double *a, 
    const double *b, unsigned int column, unsigned int row);

__global__ void vectorMultiplicationKernel(double *c, const double *a, const double *b, unsigned int column)
{
    int i = threadIdx.x;
    double result = 0;
    for (int j = 0; j < column; j++) {
        result += a[i * column + j] * b[j];
    }
    c[i] = result;
}

int main()
{
    const int row = 3;
    const int column = 5;
    const double W[row * column] = { 
        1, 2, 3, 4, 5,
        2, 4, 6, 8, 10,
        10, 20, 30, 40, 50
    };
    const double x[column] = { 10, 10, 10, 10, 10 };
    double y[row] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = vectorMatrixMulWithCuda(y, W, x, column, row);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "vectorMultiplicationWithCuda failed!");
        return 1;
    }

    printf("{\n 1,  2,  3,  4,  5,\n 2,  4,  6,  8,  10,\n 10, 20, 30, 40, 50\n} * \n{10, 10, 10, 10, 10}\n = {%0.4f, %0.4f, %0.4f}\n",
        y[0], y[1], y[2]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t vectorMatrixMulWithCuda(double *c, const double *a, const double *b, unsigned int column, unsigned int row)
{
    double *dev_a = 0;
    double *dev_b = 0;
    double *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, row * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, row * column * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, column * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, row * column * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, column * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    vectorMultiplicationKernel<<<1, row>>>(dev_c, dev_a, dev_b, column);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, row * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

