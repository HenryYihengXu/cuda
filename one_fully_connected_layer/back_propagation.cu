#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t backPropagation(double *out, double *x, 
    double *y, double *W, unsigned int column, unsigned int row, double eta);

__global__ void subtractKernel(double *out, double *y)
{
    int i = threadIdx.x;
    out[i] -= y[i];
}

__global__ void updateWKernel(double *out, double *x, double *W, int column, double eta)
{
    int i = threadIdx.x;
    W[i] = W[i] - eta * out[i / column] * x[i % column];
}


int main()
{
    int row = 3;
    int column = 5;
    double W[row * column] = { 
        1, 2, 3, 4, 5,
        2, 4, 6, 8, 10,
        10, 20, 30, 40, 50
    };
    double x[column] = { 10, 10, 10, 10, 10 };
    double out[column] = {150, 300, 1500};
    double y[row] = { 140, 310, 1600 };
    double eta = 0.01;

    // Add vectors in parallel.
    cudaError_t cudaStatus = backPropagation(out, x, y, W, column, row, eta);
    printf("new W = \n{ \n %.4f, %.4f, %.4f, %.4f, %.4f\n %.4f, %.4f, %.4f, %.4f, %.4f\n %.4f, %.4f, %.4f, %.4f, %.4f\n}\n",
        W[0], W[1], W[2], W[3], W[4], W[5], W[6], W[7], W[8], W[9], W[10], W[11], W[12], W[13], W[14]);

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
    double *y, double *W, unsigned int column, unsigned int row, double eta)
{
    double *dev_y = 0;
    double *dev_x = 0;
    double *dev_out = 0;
    double *dev_W = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&dev_y, row * sizeof(double));
    cudaStatus = cudaMalloc((void**)&dev_x, column * sizeof(double));
    cudaStatus = cudaMalloc((void**)&dev_out, row * sizeof(double));
    cudaStatus = cudaMalloc((void**)&dev_W, row * column * sizeof(double));

    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(dev_y, y, row * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x, x, column * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_out, out, row * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_W, W, row * column * sizeof(double), cudaMemcpyHostToDevice);
    
    
    // Compute (out - y) to get the differential of cost on predictions
    subtractKernel<<<1, column>>>(dev_out, dev_y);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Compute the differential of cost on weights and update weights: W = W - eta*(delta*(x)T)
    updateWKernel<<<1, column * row>>>(dev_out, dev_x, dev_W, column, eta);

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
    cudaFree(dev_y);
    cudaFree(dev_x);
    cudaFree(dev_W);
    
    return cudaStatus;
}

