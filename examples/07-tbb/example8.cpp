// =================================================================
//
// File: example7.cpp
// Author: Pedro Perez
// Description: This file contains the code that implements the
//				enumeration sort algorithm using Intel's TBB.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include "utils.h"

const int SIZE = 100000;

using namespace std;
using namespace tbb;

class EnumSort {
	private:
		int *a, *result, *pos;

	public:
		EnumSort(int *a, int *r) : a(a), result(r) { pos = new int[SIZE]; }

    void operator() (const blocked_range<int> &r) const {
      for (int i = r.begin(); i < r.end(); i++) {
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
};


int main(int argc, char* argv[]) {
	int i, *a, *result, *pos;
	double ms;

	a = (int*) malloc(sizeof(int) * SIZE);
	result = (int*) malloc(sizeof(int) * SIZE);
	random_array(a, SIZE);
	display_array("before", a);

	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
    EnumSort juanito_fast_a_f_boi(a, result);
		start_timer();

    parallel_for(blocked_range<int>(0, SIZE), juanito_fast_a_f_boi);

		ms += stop_timer();
	}
	display_array("after", result);
	printf("avg time = %.5lf ms\n", (ms / N));

  delete [] a;
  delete [] result;
	return 0;
}
