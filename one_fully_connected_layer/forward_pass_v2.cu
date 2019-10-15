#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t vectorMatrixMulWithCuda(int *c, const int *a, 
    const int *b, unsigned int column, unsigned int row);

__global__ void addKernel(int *d, int *c, unsigned int column)
{
    int i = threadIdx.x;
    int result = 0;
    for (int j = 0; j < column; j++) {
        result += c[i * column + j];
    }
    d[i] = result;
}

__global__ void mulKernel(int *c, const int *a, const int *b, unsigned int column)
{
    int i = threadIdx.x;
    c[i] = a[i] * b[i / column];
}

int main()
{
    const int row = 3;
    const int column = 5;
    const int a[row * column] = { 
        1, 2, 3, 4, 5,
        2, 4, 6, 8, 10,
        10, 20, 30, 40, 50
    };
    const int b[column] = { 10, 10, 10, 10, 10 };
    int d[row] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = vectorMatrixMulWithCuda(d, a, b, column, row);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "vectorMultiplicationWithCuda failed!");
        return 1;
    }

    printf("{\n 1,  2,  3,  4,  5,\n 2,  4,  6,  8,  10,\n 10, 20, 30, 40, 50\n} * \n{10, 10, 10, 10, 10}\n = {%d, %d, %d}\n",
        d[0], d[1], d[2]);

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
cudaError_t vectorMatrixMulWithCuda(int *d, const int *a, const int *b, unsigned int column, unsigned int row)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    int *dev_d = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, row * column * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, row * column * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, column * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_d, row * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, row * column * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, column * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    mulKernel<<<1, row * column>>>(dev_c, dev_a, dev_b, column);

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

     // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, row>>>(dev_d, dev_c, column);

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
    cudaStatus = cudaMemcpy(d, dev_d, row * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_d);
    
    return cudaStatus;
}
