#include "book.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

__global__ void matrixMul(double *a. double *b, double *c, int p) {
    int tid = blockIdx.x * gridDim.x + blockIdx.y;
    for (int i = 0; i < p; i++) {
        c[tid] = c[tid] + a[blockIdx.x * p + i] * b[i * gridDim.y + blockIdx.y];
    }
}

int main(int argc, char *argv[]) {
    int m = atoi(argv[1]);
    int p = atoi(argv[2]);
    int n = atoi(argv[3]);

    double a[m * p];
    double b[p * n];
    double c[m * n];
    double *dev_a, *dev_b, dev_c;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            a[i * p + j] = 1;
        }
    }

    for (int i = 0; i < p; i++) {
        for (int j = 0; j < n; j++) {
            b[i * n + j] = 1;
        }
    }

    HANDLE_ERROR( cudaMalloc( (void**)&dev_a, m * p * sizeof(double) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b, p * n * sizeof(double) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a, m * n * sizeof(double) ) );

    HANDLE_ERROR(cudaMemcpy(dev_a, a, m * p * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, p * n * sizeof(double), cudaMemcpyHostToDevice));

    dim3 grid(m, n);
    matrixMul<<<grid, 1>>>(dev_a, dev_b, dev_c, p);

    HANDLE_ERROR(cudaMemcpy(c, dev_c, m * n * sizeof(double), cudaMemcpyDeviceToHost));

    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("c[%d][%d] = %f\n", i, j, c[i * n + j]);
    //     }
    // }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}