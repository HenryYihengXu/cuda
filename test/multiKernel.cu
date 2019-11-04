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
    int kernelNum = atoi(argv[3]);

    double a[kernelNum][blockNum * threadNum];
    double *dev_a[kernelNum];
    cudaStream_t streams[kernelNum];

    for(int i = 0; i < kernelNum; i++) {
        HANDLE_ERROR( cudaMalloc( (void**)&dev_a[i], blockNum * threadNum * sizeof(double) ) );
    }
    for(int i = 0; i < kernelNum; i++) {
        cudaStreamCreate(&streams[i]);
    }
    for(int i = 0; i < kernelNum; i++) {
        doSomeComputation<<<blockNum, threadNum, 0, streams[i]>>>(dev_a[i]);
    }   
    
    for(int i = 0; i < kernelNum; i++) {
        HANDLE_ERROR(cudaMemcpy(a[i], dev_a[i], blockNum * threadNum * sizeof(double), cudaMemcpyDeviceToHost));
    } 

    // for (int i = 0; i < kernelNum; i++) {
    //     printf("kernel %d:\n", i);
    //     for (int j = 0; j < blockNum * threadNum; j++) {
    //         printf("a[%d] = %f\n", j, a[j]);
    //     }
    //     printf("\n");
    // }
   
    for(int i = 0; i < kernelNum; i++) {
        cudaFree(dev_a[i]);
    } 
    
    return 0;
}