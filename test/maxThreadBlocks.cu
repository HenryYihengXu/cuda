#include "book.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

__global__ void doSomeComputation(double *a) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double result = 0;
    for (int i = 0; i < 10000; i++) {
        result += sqrt(pow(3.14159,-i));
    }
    a[tid] = result;
}

int main(int argc, char *argv[]) {
    int blockNum = atoi(argv[1]);
    int threadNum = atoi(argv[2]);

    double a[blockNum * threadNum];
    double *dev_a;
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a, blockNum * threadNum * sizeof(double) ) );

    doSomeComputation<<<blockNum, threadNum>>>(dev_a);

    HANDLE_ERROR(cudaMemcpy(a, dev_a, blockNum * threadNum * sizeof(double), cudaMemcpyDeviceToHost));

    // for (int i = 0; i < blockNum * threadNum; i++) {
    //     printf("a[%d] = %f\n", i, a[i]);
    // }

    cudaFree(dev_a);

    return 0;
}