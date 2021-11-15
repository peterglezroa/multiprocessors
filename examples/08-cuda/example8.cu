// =================================================================
//
// File: example8.cu
// Author(s):
// Description: This file contains the code that implements the
//				enumeration sort algorithm using CUDA.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "utils.h"

#define SIZE 100000
#define THREADS 256
#define BLOCKS	MMIN(32, ((SIZE / THREADS) + 1))

__global__ void sort(int *arr, int *res) {
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	for (int i = tid; i < SIZE; i += blockDim.x * gridDim.x) {
		int n = 0;
		for(int j = 0; j < SIZE; j++) {
			if (arr[i] > arr[j] || arr[i] == arr[j] && i > j) {
				n++;
			}
		}
		res[n] = arr[i];
	}
}

int main() {
	int *arr, *res;
	int *d_arr, *d_res;

	arr = (int *)malloc(sizeof(int) * SIZE);
	res = (int *)malloc(sizeof(int) * SIZE);
	cudaMalloc((void**) &d_arr, sizeof(int) * SIZE);
	cudaMalloc((void**) &d_res, sizeof(int) * SIZE);

	printf("Starting...\n");
	double ms = 0;
	for (int i = 0; i < N; i++) {
		random_array(arr, SIZE);
		cudaMemcpy(d_arr, arr, sizeof(int) * SIZE, cudaMemcpyHostToDevice);
		start_timer();

		sort<<<BLOCKS, THREADS>>>(d_arr, d_res);
		ms += stop_timer();
	}
	cudaMemcpy(res, d_res, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);
	display_array("after", res);
	printf("avg time = %.5lf ms\n", (ms / N));
	cudaFree(d_arr);
	cudaFree(d_res);
	free(arr);
	free(res);
	return 0;
}
