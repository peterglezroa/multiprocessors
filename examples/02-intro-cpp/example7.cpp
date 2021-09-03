// =================================================================
//
// File: example7.cpp
// Author: Pedro Perez
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

#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "utils.h"

#define MAXIMUM 1000000 //1e6

using namespace std;

class PrimeCalculator {
    private:
        int *array, size;
    public:
        PrimeCalculator(int *array, int size) : array(array), size(size) {}

        int * getPrimes() const { return array; }

        void calculate() {
            for (int x = 2; x < size; x++) {
                if (array[x] != 0) {
                    array[x] = 1;
                    for (int j = 0; j < sqrt((double)x); j++)
                        if (x % j == 0)
                            array[x] = 0;

                    if (array[x] == 1)
                        for (int j = 0; x*j < N; j++)
                            array[x*j] = 0;
                }
            }
        }
};

int main(int argc, char* argv[]) {
	int i, *a;
	double ms;

	a = new int[MAXIMUM + 1];
	cout << "At first, neither is a prime. We will display to TOP_VALUE:\n";
    cout << MAXIMUM << TOP_VALUE;
	for (i = 2; i < TOP_VALUE; i++) {
        a[i] = -1;
		cout << i << " ";
	}
	cout << "\n";

	cout << "Starting..." << endl;
	ms = 0;
    cout << "debug 1";
    cout << "debug 2";

	// create object here
    PrimeCalculator calc = PrimeCalculator(a, MAXIMUM+1);

    start_timer();
    calc.calculate();
    ms += stop_timer();

	cout << "Expanding the numbers that are prime to TOP_VALUE:\n";
	for (i = 2; i < TOP_VALUE; i++) {
		if (a[i] == 1) {
			printf("%i ", i);
		}
	}
	cout << "avg time = " << setprecision(15) << (ms / N) << " ms" << endl;

	delete [] a;
	return 0;
}
