// =================================================================
//
// File: Example8.java
// Author: Pedro González A01651517, Juan Alcantara A01703947
// Description: This file implements the enumeration sort algorithm.
// 				The time this implementation takes will be used as the
//				basis to calculate the improvement obtained with
//				parallel technologies.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.util.Arrays;

public class Example8 {
	private static final int SIZE = 100_000;
	private int array[];

	public Example8(int array[]) {
		this.array = array;
	}

	public void doTask() {
        int n = 0;
        int[] res = new int[SIZE];
        for (int i = 0; i < SIZE; i++) {
            n = 0;
            for (int j = 0; j < SIZE; j++)
                if (array[i] > array[j] || array[i] == array[j] && i > j)
                    n++;
            res[n] = array[i];
        }
        array = res;
	}

	public int[] getSortedArray() {
		return array;
	}

	public static void main(String args[]) {
		int array[] = new int[SIZE];
		long startTime, stopTime;
		double ms;
		Example8 obj = null;

		Utils.randomArray(array);
		Utils.displayArray("before", array);

        obj = new Example8(array);
		System.out.printf("Starting...\n");
		ms = 0;
		for (int i = 0; i < 2; i++) {
			startTime = System.currentTimeMillis();

			// pace your code here.
            obj.doTask();

			stopTime = System.currentTimeMillis();

			ms += (stopTime - startTime);
		}
		Utils.displayArray("after", obj.getSortedArray());
		System.out.printf("avg time = %.5f\n", (ms / Utils.N));
	}
}
