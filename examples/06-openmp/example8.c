// =================================================================
//
// File: example8.c
// Author(s):
// Description: This file contains the code that implements the
//				enumeration sort algorithm using OpenMP.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

#define SIZE 100000

void enum_sort(int* a, int* pos, int* result) {
  #pragma omp parallel for shared(result)
	for (int i = 0; i < SIZE; i++) {
		pos[i] = 0;
		for (int j = 0; j < SIZE; j++) {
			if (a[i] > a[j] || a[i] == a[j] && i > j) {
				pos[i] += 1;
			}
		}
	}
	for (int i = 0; i < SIZE; i++)
		result[pos[i]] = a[i];
}

int main(int argc, char* argv[]) {
	int i, *a, *result, *pos;
	double ms;

	a = (int*) malloc(sizeof(int) * SIZE);
	pos = (int*) malloc(sizeof(int) * SIZE);
	result = (int*) malloc(sizeof(int) * SIZE);
	random_array(a, SIZE);
	display_array("before", a);

	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();

		enum_sort(a, pos, result);

		ms += stop_timer();
	}
	display_array("after", result);
	printf("avg time = %.5lf ms\n", (ms / N));

	free(a);
	free(pos);
	free(result);
	return 0;
}
