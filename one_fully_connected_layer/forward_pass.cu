#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

cudaError_t forwardPass(double *x, double *y,
    double *W, int row, int column);

__global__ void vectorMultiplicationKernel(double *x, double *y, double *W,
    int row, int column)
{
    int tid = blockIdx.x;
    if (tid >= row) {
        return;
    }
    double result = 0;
    for (int j = 0; j < column; j++) {
        result += W[tid * column + j] * x[j];
    }
    for (int j = 0; j < 10000; j++) {
        for (int k = 0; k < 10000; k++) {
            result++;
            result--;
        }
    }
    y[tid] = result;
}

int main(int argc, char *argv[])
{
    int row = atoi(argv[1]);
    int column = atoi(argv[2]);

    double *W = (double*)malloc(row * column * sizeof(double));
    double *x = (double*)malloc(column * sizeof(double));
    double *y = (double*)malloc(row * sizeof(double));

    for (int i = 0; i < column; i++) {
        x[i] = 10;
    }

    for (int i = 0; i < row * column; i++) {
        W[i] = 10;
    }

    cudaError_t cudaStatus = forwardPass(x, y, W, row, column);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "vectorMultiplicationWithCuda failed!");
        return 1;
    }

    for (int i = 0; i < row; i++) {
        printf("%.2f ", y[i]);
    }
    printf("\n");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t forwardPass(double *x, double *y, double *W,
    int row, int column)
{
    double *dev_x = 0;
    double *dev_y = 0;
    double *dev_W = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_x, column * sizeof(double));
    cudaStatus = cudaMalloc((void**)&dev_y, row * sizeof(double));
    cudaStatus = cudaMalloc((void**)&dev_W, row * column * sizeof(double));

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_x, x, column * sizeof(double), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_W, W, row * column * sizeof(double), cudaMemcpyHostToDevice);

    // Launch a kernel on the GPU with one thread for each element.
    vectorMultiplicationKernel<<<row, 1>>>(dev_x, dev_y, dev_W, row, column);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "vectorMultiplicationKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
    cudaStatus = cudaMemcpy(y, dev_y, row * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_W);

    return cudaStatus;
}
