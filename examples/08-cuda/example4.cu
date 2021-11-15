// =================================================================
//
// File: example4.cu
// Author(s):
//					A01651517 Pedro Luis González Roa
//					A01703947 Juan Alejandro Alcántara Minaya
// Description: This file contains the code to count the number of
//				even numbers within an array using CUDA.
//
// =================================================================

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils.h"

#define SIZE 1000000000
#define THREADS	256
#define BLOCKS	MMIN(32, ((SIZE / THREADS) + 1))

__device__ int is_even(int x) { return !(x % 2); }

__global__ void array_even(int *a, int *r) {
    int tid = threadIdx.x + (blockIdx.x*blockDim.x);

    // Recycle threads
    while (tid < SIZE) {
        r[tid] += is_even(a[tid]);
        tid += blockDim.x * gridDim.x;
    }
}

int main(int argc, char* argv[]) {
    int i, *a, *r, *a_gpu, *r_gpu;
    long result = 0;
    double ms;

    a = (int *)malloc(sizeof(int) * SIZE);
    r = (int *)malloc(sizeof(int) * SIZE);
    fill_array(a, SIZE);
    display_array("a", a);

    // Copy to gpu 
    cudaMalloc((void**) &a_gpu, sizeof(int)*SIZE);
    cudaMemcpy(a_gpu, a, sizeof(int)*SIZE, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &r_gpu, sizeof(int)*SIZE);

    printf("Starting...\n");
    ms = 0;
    for (int i = 0; i < N; i++) {
        start_timer();

        array_even<<<BLOCKS, THREADS>>>(a_gpu, r_gpu);

        ms += stop_timer();
    }

    cudaMemcpy(r, r_gpu, sizeof(int)*SIZE, cudaMemcpyDeviceToHost);
    
    for (i = 0; i < SIZE; i++) result += r[i];

	printf("result = %li\n", result);
	printf("avg time = %.5lf ms\n", (ms / N));

    cudaFree(r_gpu); cudaFree(a_gpu);
    free(a); free(r);
    return 0;
}
