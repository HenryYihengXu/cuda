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

    // Allocate GPU buffers for three vectors (two input, one output)    .
    
    cudaMalloc((void**)&dev_x1, column * sizeof(double));
    cudaMalloc((void**)&dev_y1, row * sizeof(double));
    cudaMalloc((void**)&dev_W1, row * column * sizeof(double));
    cudaMemcpy(dev_x1, x1, column * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_W1, W1, row * column * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_x2, column * sizeof(double));
    cudaMalloc((void**)&dev_y2, row * sizeof(double));
    cudaMalloc((void**)&dev_W2, row * column * sizeof(double));
    cudaMemcpy(dev_x2, x2, column * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_W2, W2, row * column * sizeof(double), cudaMemcpyHostToDevice);

    //Launch a kernel on the GPU with one thread for each element.
    cudaStream_t s1;
    cudaStream_t s2;
    cudaStreamCreate(&s1);
    vectorMultiplicationKernel<<<row, 1, 0, s1>>>(dev_x1, dev_y1, dev_W1, row, column);

    cudaStreamCreate(&s2);
    vectorMultiplicationKernel<<<row, 1, 0, s2>>>(dev_x2, dev_y2, dev_W2, row, column);
    // vectorMultiplicationKernel<<<row, 1>>>(dev_x1, dev_y1, dev_W1, row, column);
    // vectorMultiplicationKernel<<<row, 1>>>(dev_x2, dev_y2, dev_W2, row, column);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(y1, dev_y1, row * sizeof(double), cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(y2, dev_y2, row * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dev_x1);
    cudaFree(dev_y1);
    cudaFree(dev_W1);
    cudaFree(dev_x2);
    cudaFree(dev_y2);
    cudaFree(dev_W2);

    return cudaStatus;
}


