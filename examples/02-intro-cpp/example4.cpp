// =================================================================
//
// File: example2.cpp
// Author(s): Pedro González A01651517, Juan Alcantara A01703947
// Description: This file contains the code to count the number of
//				even numbers within an array. The time this implementation
//				takes will be used as the basis to calculate the
//				improvement obtained with parallel technologies.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <climits>
#include <algorithm>
#include "utils.h"

const int SIZE = 1000000000; //1e9

using namespace std;

// implement your class here
class EvenCalculator {
    private:
        int *array, size, result;
    public:
        EvenCalculator (int *array, int size) : array(array), size(size) {}

        int getResult() const { return result; }

        void calculate() {
            result = 0;
            for (int i = 0; i < size; i++)
                if ( !(array[i] % 2) )
                    result++;
        }
};

int main(int argc, char* argv[]) {
	int *a;
	double ms;

	a = new int[SIZE];
	fill_array(a, SIZE);
	display_array("a", a);

	cout << "Starting..." << endl;
	ms = 0;

	// create object here
    EvenCalculator calc = EvenCalculator(a, SIZE);

    start_timer();
    calc.calculate();
    ms += stop_timer();

	cout << "result = " << calc.getResult() << endl;
	cout << "avg time = " << setprecision(15) << (ms / N) << " ms" << endl;

	delete [] a;
	return 0;
}
