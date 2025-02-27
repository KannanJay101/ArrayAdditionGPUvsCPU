#include <cuda_runtime.h>  
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>
#include <chrono>
#include <iostream>
using namespace std;

#define N (1 << 22) // 4,194,304 elements or 2^22 elements

// GPU function to sum 3 arrays
__global__ void three_array_sum_GPU(int *a, int *b, int *c, int *d, int size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        d[gid] = a[gid] + b[gid] + c[gid];
    }
}

void three_array_sum_CPU(int *a, int *b, int *c, int *d, int size) {
    for (int i = 0; i < size; i++) {
        d[i] = a[i] + b[i] + c[i];
    }
}

int main() {
    int size = N; // Number of elements in the array
    int block_size = 128; // Number of threads in a block

    int NO_BYTES = size * sizeof(int); // number of bytes : size (10000) * size of int (4 bytes)
    
    int *h_a, *h_b, *h_c, *h_d, *gpu_results; 

    // Allocating memory for each of these pointers 
    h_a = (int*)malloc(NO_BYTES);
    h_b = (int*)malloc(NO_BYTES);
    h_c = (int*)malloc(NO_BYTES);
    h_d = (int*)malloc(NO_BYTES);
    gpu_results = (int*)malloc(NO_BYTES);

    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        h_a[i] = rand() % size;
        h_b[i] = rand() % size;
        h_c[i] = rand() % size;
    }

    // CPU function to sum 3 arrays
    auto cpu_start = std::chrono::high_resolution_clock::now();
    three_array_sum_CPU(h_a, h_b, h_c, h_d, size);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU time (s): " << cpu_duration.count() << std::endl;
    
    memset(gpu_results, 0, NO_BYTES);

    // Device Pointer
    int *d_a, *d_b, *d_c, *d_d;

    // Allocating memory on the device the VRAM
    cudaMalloc((void**)&d_a, size * sizeof(int));
    cudaMalloc((void**)&d_b, size * sizeof(int));
    cudaMalloc((void**)&d_c, size * sizeof(int));
    cudaMalloc((void**)&d_d, size * sizeof(int));

    cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, NO_BYTES, cudaMemcpyHostToDevice);

    dim3 block(block_size);    // 128 threads per block
    dim3 grid((size + block.x - 1) / block.x); // Calculate grid size

    auto gpu_start = std::chrono::high_resolution_clock::now();
    three_array_sum_GPU<<<grid, block>>>(d_a, d_b, d_c, d_d, size);
    cudaDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_duration = gpu_end - gpu_start;
    std::cout << "GPU time (s): " << gpu_duration.count() << std::endl;

    cudaMemcpy(gpu_results, d_d, NO_BYTES, cudaMemcpyDeviceToHost); // Copy the results from device to host

    // Print statement to verify code execution
    std::cout << "GPU computation completed." << std::endl;

    //free the memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    
    free(h_a);
    free(h_b);
    free(h_c);  
    free(h_d);
    free(gpu_results);

    return 0;
}

// Compile command: nvcc -o exercises3 exercises3.cu
// Run command: ./exercises3
