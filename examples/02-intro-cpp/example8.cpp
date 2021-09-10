// =================================================================
//
// File: example5.cpp
// Author: Pedro González A01651517, Juan Alcantara A01703947
// Description: This file contains the code that implements the
//				enumeration sort algorithm. The time this implementation
//				takes ill be used as the basis to calculate the
//				improvement obtained with parallel technologies.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <cstring>
#include "utils.h"

const int SIZE = 10000; //1e4

using namespace std;

// implement your class here
class EnumerationSort {
	private:
		int *a, *pos, *result;
	
	public:
		EnumerationSort (
				int *a, int *pos, int *result
		) : a(a), pos(pos), result(result) {}

		void enum_sort() {
			for (int i = 0; i < SIZE; i++) {
				pos[i] = 0;
				for (int j = 0; j < SIZE; j++) {
					if (a[i] > a[j] || a[i] == a[j] && i > j) {
						pos[i] += 1;
					}
				}
			}
			for (int i = 0; i < SIZE; i++) {
				result[pos[i]] = a[i];
			}
		}
};

int main(int argc, char* argv[]) {
	int *a, *pos, *result;
	double ms;

	a = new int[SIZE];
	pos = new int [SIZE];
	result = new int [SIZE];
	random_array(a, SIZE);
	display_array("before", a);

	cout << "Starting..." << endl;
	ms = 0;
	// create object here
	EnumerationSort enumSorter = EnumerationSort(a, pos, result);
	for (int i = 0; i < N; i++) {
		start_timer();

		// call your method here.
		enumSorter.enum_sort();

		ms += stop_timer();
	}

	display_array("after", result);
	cout << "avg time = " << setprecision(15) << (ms / N) << " ms" << endl;

	delete [] a;
	delete [] pos;
	delete [] result;
	return 0;
}
