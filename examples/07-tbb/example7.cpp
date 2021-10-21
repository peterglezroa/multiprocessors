// =================================================================
//
// File: example7.cpp
// Authors:
//        A01651517 Pedro González
//				A01703947 Juan Alcántara
// Description: This file contains the code to brute-force all
//				prime numbers less than MAXIMUM using Intel's TBB.
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

const int MAXIMUM = 1000000; //1e6

using namespace std;
using namespace tbb;

// place your code here
void is_prime(int* a, int x) {
	if (a[x] != 0) { // Check if it has not been already set as no prime
		a[x] = 1; // Define as prime until proven wrong

    // Calculate if prime
		for (int i = 2; i < sqrt((double)x); i++)
			if (x % i == 0)
				a[x] = 0;

    // If prime then all the multiples are not prime
		if (a[x] == 1)
			for (int i = 2; x * i < N; i++)
				a[x * i] = 0;
	}
}
class PrimeNumber {
	private:
		int* arr;
	
	public:
		PrimeNumber(int *a) : arr(a) {}

	void operator() (const blocked_range<int> &r) const {
		for (int i = r.begin(); i != r.end(); i++) {
			is_prime(arr, i);
		}
	}
};

int main (int argc, char* argv[]) {
	int *a;
	double ms;
	cout << "Starting..." << endl;
	ms = 0;

	a = new int[MAXIMUM];

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < MAXIMUM; j++)
			a[j] = -1;

		PrimeNumber identifier(a);
		start_timer();
		parallel_for(blocked_range<int>(2, MAXIMUM), identifier);

		ms += stop_timer();
	}

	for (int i = 2; i < N; i++) {
		if (a[i] == 1) {
			cout << i << " ";
		}
	}
	cout << "avg time = " << setprecision(15) << (ms / N) << " ms" << endl;

	delete [] a;
	return 0;
}
