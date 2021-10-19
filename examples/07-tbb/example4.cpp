// =================================================================
//
// File: Example4.cpp
// Authors:
//				 A01651517 Pedro González
//				 A01703947 Juan Alcántara
// Description: This file contains the code to count the number of
//				even numbers within an array using Intel's TBB.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include "utils.h"

#define SIZE 1000000000 //1e9

using namespace std;
using namespace tbb;

class CountEven {
	private:
		int *array, result;
	public:
		CountEven(int *arr) : array(arr), result(0) {}

		CountEven(CountEven &x, split) : array(x.array), result(0) {}

		int getResult() const {
			return result;
		}

		void operator() (const blocked_range<int> &r) {
			for (int i = r.begin(); i != r.end(); i++) {
				result += !(array[i] % 2);
			}
		}

		void join(const CountEven &x) {
			result += x.result;
		}
};

int main(int argc, char* argv[]) {
	int *a, pos, result;
	double ms;

	a = new int[SIZE];
	random_array(a, SIZE);
	display_array("a", a);

	srand(time(0));
	pos = rand() % SIZE;
	printf("Setting value 0 at %i\n", pos);
	a[pos] = 0;

	cout << "Starting..." << endl;
	ms = 0;

	for (int i = 0; i < N; i++) {
		start_timer();

		CountEven counter(a);
		parallel_reduce(blocked_range<int>(0, SIZE), counter);
		result = counter.getResult();

		ms += stop_timer();
	}
	cout << "result = " << result << endl;
	cout << "avg time = " << setprecision(15) << (ms / N) << " ms" << endl;

	delete [] a;
	return 0;
}

