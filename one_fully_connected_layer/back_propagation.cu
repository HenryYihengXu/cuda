#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

cudaError_t backPropagation(double *out, double *x, 
    double *y, double *W, unsigned int row, unsigned int column, double eta);

__global__ void subtractKernel(double *out, double *y, unsigned int row)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= row) {
        return;
    }
    out[thread_idx] -= y[thread_idx];
}

__global__ void updateWKernel(double *out, double *x, double *W, 
    unsigned int row, unsigned int column, double eta)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= row * column) {
        return;
    }
    W[thread_idx] = W[thread_idx] - eta * out[thread_idx / column] * x[thread_idx % column];
}


int main(int argc, char *argv[])
{
    unsigned int row = atoi(argv[1]);
    unsigned int column = atoi(argv[2]);
    double eta = atof(argv[3]);
    double *W = (double*)malloc(row * column * sizeof(double));
    double *x = (double*)malloc(column * sizeof(double));
    double *y = (double*)malloc(row * sizeof(double));
    double *out = (double*)malloc(row * sizeof(double));

    for (int i = 0; i < column; i++) {
        x[i] = 10;
    }

    for (int i = 0; i < row * column; i++) {
        W[i] = 10;
    }

    for (int i = 0; i < column; i++) {
        y[i] = 10;
    }

    for (int i = 0; i < column; i++) {
        out[i] = 11;
    }

    // Add vectors in parallel.
    cudaError_t cudaStatus = backPropagation(out, x, y, W, row, column, eta);
    
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            printf("%.2f ", W[i * column + j]);
        }
        printf("\n");
    }
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
cudaError_t backPropagation(double *out, double *x, 
    double *y, double *W, unsigned int row, unsigned int column, double eta)
{
    double *dev_out = 0;
    double *dev_x = 0;
    double *dev_y = 0;
    double *dev_W = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&dev_out, row * sizeof(double));
    cudaStatus = cudaMalloc((void**)&dev_x, column * sizeof(double));
    cudaStatus = cudaMalloc((void**)&dev_y, row * sizeof(double));
    cudaStatus = cudaMalloc((void**)&dev_W, row * column * sizeof(double));

    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(dev_out, out, row * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x, x, column * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, row * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_W, W, row * column * sizeof(double), cudaMemcpyHostToDevice);
    
    // Compute (out - y) to get the differential of cost on predictions
    subtractKernel<<<column / 512 + 1, 512>>>(dev_out, dev_y, row);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Compute the differential of cost on weights and update weights: W = W - eta*(delta*(x)T)
    updateWKernel<<<row * column / 512 + 1, 512>>>(dev_out, dev_x, dev_W, row, column, eta);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(W, dev_W, row * column * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_out);
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_W);
    
    return cudaStatus;
}

