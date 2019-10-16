#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t vectorMatrixMulWithCuda(double *c,  double *a, 
     double *b, unsigned int column, unsigned int row);
    
void size_3x5_test();
void size_10x10_test();

__global__ void addKernel(double *d, double *c, unsigned int column)
{
    int i = threadIdx.x;
    double result = 0;
    for (int j = 0; j < column; j++) {
        result += c[i * column + j];
    }
    d[i] = result;
}

__global__ void mulKernel(double *c,  double *a,  double *b, unsigned int column)
{
    int i = threadIdx.x;
    c[i] = a[i] * b[i / column];
}

int main()
{
    size_3x5_test();
    size_10x10_test();

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t vectorMatrixMulWithCuda(double *d,  double *a,  double *b, unsigned int column, unsigned int row)
{
    double *dev_a = 0;
    double *dev_b = 0;
    double *dev_c = 0;
    double *dev_d = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, row * column * sizeof(double));
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

    cudaStatus = cudaMalloc((void**)&dev_d, row * sizeof(double));
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
    mulKernel<<<1, row * column>>>(dev_c, dev_a, dev_b, column);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "mulKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
    cudaStatus = cudaMemcpy(d, dev_d, row * sizeof(double), cudaMemcpyDeviceToHost);
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

void size_3x5_test() {
    int row = 3;
    int column = 5;
    double W[row * column] = { 
        1, 2, 3, 4, 5,
        2, 4, 6, 8, 10,
        10, 20, 30, 40, 50
    };
    double x[column] = { 10, 10, 10, 10, 10 };
    double y[row] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = vectorMatrixMulWithCuda(y, W, x, column, row);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "vectorMultiplicationWithCuda failed!");
        return;
    }

    printf("{\n 1,  2,  3,  4,  5,\n 2,  4,  6,  8,  10,\n 10, 20, 30, 40, 50\n} * \n{10, 10, 10, 10, 10}\n = {%.2f, %.2f, %.2f}\n",
        y[0], y[1], y[2]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return;
    }
}

void size_10x10_test() {
    int row = 10;
    int column = 10;
    double W[row * column] = {0};
    double x[column] = {0};
    double y[row] = { 0 };

    for (int i = 0; i < row * column; i++) {
        W[i] = 10;
    }

    for (int i = 0; i < column; i++) {
        x[i] = 10;
    }

    // Add vectors in parallel.
    cudaError_t cudaStatus = vectorMatrixMulWithCuda(y, W, x, column, row);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "vectorMultiplicationWithCuda failed!");
        return;
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
        return;
    }
}


