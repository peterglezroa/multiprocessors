// =================================================================
//
// File: example7.cu
// Author(s):
//					A01651517 Pedro Luis González Roa
//					A01703947 Juan Alejandro Alcántara Minaya
// Description: This file contains the code to brute-force all
//				prime numbers less than MAXIMUM using CUDA.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "utils.h"

#define MAXIMUM 1000000 //1e6
#define THREADS 256
#define BLOCKS	MMIN(32, ((SIZE / THREADS) + 1))

__global__ void is_prime(int*a) {
}

int main(int argc, char* argv[]) {
    int i, *a, *a_gpu;
    double ms;

    // Memory in cpu
    a = (int *)malloc(sizeof(int) * (MAXIMUM+1));

    // Memory in gpu
    cudaMalloc((void**) &a_gpu, sizeof(int)*MAXIMUM+1);

    printf("Starting...\n");
    ms = 0;

    for (i = 0; i < N; i++) {
        start_timer();

        is_prime<<<BLOCKS, THREADS>>>(a_gpu);

        ms += stop_timer();
    }

    // Copy back
    cudaMemcpy(a, a_gpu, sizeof(int)*SIZE, cudaMemcpyDeviceToHost);

    printf("Expanding the numbers that are prime to TOP_VALUE:\n");
    for (i = 2; i < TOP_VALUE; i++)
        if (a[i]) printf("%i ", i);
	printf("\n");
	printf("avg time = %.5lf ms\n", (ms / N));

    cudaFree(a_gpu);
    free(a);
    return 0;
}
