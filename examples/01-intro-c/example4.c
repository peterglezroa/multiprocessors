// =================================================================
//
// File: example4.c
// Author(s):
//					A01651517 Pedro Luis González Roa
//					A01703947 Juan Alejandro Alcántara Minaya
// Description: This file contains the code to count the number of
//				even numbers within an array. The time this implementation
//				takes will be used as the basis to calculate the
//				improvement obtained with parallel technologies.
//
// =================================================================

#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

// array size
#define SIZE 1000000000

// implement your code
int is_even(int x) { return !(x % 2); }

int main(int argc, char* argv[]) {
	int i, *a;
  long result = 0;
	double ms;

	a = (int *) malloc(sizeof(int) * SIZE);
	fill_array(a, SIZE);
	display_array("a", a);

	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();
    result = 0;

    for (int j = 0; j < SIZE; j++)
      result += is_even(a[j]);

		ms += stop_timer();
	}
	printf("result = %li\n", result);
	printf("avg time = %.5lf ms\n", (ms / N));
	// must display: result = 500000000

	free(a);
	return 0;
}
