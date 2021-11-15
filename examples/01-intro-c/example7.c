// =================================================================
//
// File: example7.c
// Author(s):
//					A01651517 Pedro Luis González Roa
//					A01703947 Juan Alejandro Alcántara Minaya
// Description: This file contains the code to brute-force all
//				prime numbers less than MAXIMUM. The time this
//				implementation takes will be used as the basis to
//				calculate the improvement obtained with parallel
//				technologies.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include <math.h>

#define MAXIMUM 100000000 //1e6

// void is_prime(int* a, int x) {
// 	for (int i = 2; i < sqrt((double)x); i++) {
// 		if (x % i == 0) {
// 			a[x] = 0;
// 			return;
// 		}
// 	}
// 	a[x] = 1;
// }

void is_prime(int* a, int x) {
	if (a[x] != 0) {
		a[x] = 1;
		for (int i = 2; i < sqrt((double)x); i++) {
			if (x % i == 0) {
				a[x] = 0;
			}
		}
		if (a[x] == 1) {
			for (int i = 2; x * i < N; i++) {
				a[x * i] = 0;
			}
		}
	}
}

int main(int argc, char* argv[]) {
	int i, *a;
	double ms;

	a = (int *) malloc(sizeof(int) * (MAXIMUM + 1));
	printf("At first, neither is a prime. We will display to TOP_VALUE:\n");
	for (i = 2; i < TOP_VALUE; i++) {
		a[i] = -1;
		if (a[i] == -1) {
			printf("%i ", i);
		}
	}
	printf("\n");

	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();

        for (int j = 0; j < MAXIMUM; j++)
          is_prime(a, j);

		ms += stop_timer();
	}

	printf("Expanding the numbers that are prime to TOP_VALUE:\n");
	for (i = 2; i < TOP_VALUE; i++) {
		if (a[i] == 1) {
			printf("%i ", i);
		}
	}
	printf("\n");
	printf("avg time = %.5lf ms\n", (ms / N));

	free(a);
	return 0;
}
