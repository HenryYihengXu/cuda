#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

cudaError_t forwardPass(double *x1, double *y1, double *W1, 
    double *x2, double *y2, double *W2, 
    int row, int column);

__global__ void vectorMultiplicationKernel(double *x, double *y, double *W,
    int row, int column)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= row) {
        return;
    }
    double result = 0;
    for (int j = 0; j < column; j++) {
        result += W[thread_idx * column + j] * x[j];
    }
    y[thread_idx] = result;
}

int main(int argc, char *argv[])
{
    int row = atoi(argv[1]);
    int column = atoi(argv[2]);

    double *W1 = (double*)malloc(row * column * sizeof(double));
    double *x1 = (double*)malloc(column * sizeof(double));
    double *y1 = (double*)malloc(row * sizeof(double));
    double *W2 = (double*)malloc(row * column * sizeof(double));
    double *x2 = (double*)malloc(column * sizeof(double));
    double *y2 = (double*)malloc(row * sizeof(double));

    for (int i = 0; i < column; i++) {
        x1[i] = 10;
    }

    for (int i = 0; i < row * column; i++) {
        W1[i] = 10;
    }

    for (int i = 0; i < column; i++) {
        x2[i] = 10;
    }

    for (int i = 0; i < row * column; i++) {
        W2[i] = 10;
    }

    cudaError_t cudaStatus = forwardPass(x1, y1, W1, x2, y2, W2, row, column);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "vectorMultiplicationWithCuda failed!");
        return 1;
    }

    for (int i = 0; i < row; i++) {
        printf("%.2f ", y1[i]);
    }
    printf("\n");

    for (int i = 0; i < row; i++) {
        printf("%.2f ", y2[i]);
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

cudaError_t forwardPass(double *x1, double *y1, double *W1,
    double *x2, double *y2, double *W2,
    int row, int column)
{
    double *dev_x1 = 0;
    double *dev_y1 = 0;
    double *dev_W1 = 0;
    double *dev_x2 = 0;
    double *dev_y2 = 0;
    double *dev_W2 = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_x1, column * sizeof(double));
    cudaStatus = cudaMalloc((void**)&dev_y1, row * sizeof(double));
    cudaStatus = cudaMalloc((void**)&dev_W1, row * column * sizeof(double));
    cudaStatus = cudaMalloc((void**)&dev_x2, column * sizeof(double));
    cudaStatus = cudaMalloc((void**)&dev_y2, row * sizeof(double));
    cudaStatus = cudaMalloc((void**)&dev_W2, row * column * sizeof(double));

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_x1, x1, column * sizeof(double), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_W1, W1, row * column * sizeof(double), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_x2, x2, column * sizeof(double), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_W2, W2, row * column * sizeof(double), cudaMemcpyHostToDevice);

    // Launch a kernel on the GPU with one thread for each element.
    vectorMultiplicationKernel<<<row / 512 + 1, 512>>>(dev_x1, dev_y1, dev_W1, row, column);
    vectorMultiplicationKernel<<<row / 512 + 1, 512>>>(dev_x2, dev_y2, dev_W2, row, column);
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
    cudaStatus = cudaMemcpy(y1, dev_y1, row * sizeof(double), cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(y2, dev_y2, row * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_x1);
    cudaFree(dev_y1);
    cudaFree(dev_W1);
    cudaFree(dev_x2);
    cudaFree(dev_y2);
    cudaFree(dev_W2);

    return cudaStatus;
}

