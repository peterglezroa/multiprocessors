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

public class Example8 extends Thread {
	private static final int SIZE = 100_000;
	private static final int MAXTHREADS = Utils.MAXTHREADS - 1;

	private int start;
	private int array[];
	private int res[];

	public Example8(int start, int array[], int res[]) {
		this.start = start;
		this.array = array;
		this.res = res;
	}

	public void run() {
		int n = 0;
		for (int i = start; i < SIZE; i += MAXTHREADS) {
			n = 0;
			for (int j = 0; j < SIZE; j++) {
				if (array[i] > array[j] || array[i] == array[j] && i > j) {
					n++;
				}
			}
			res[n] = array[i];
		}
	}
	
	// place your code here
	public static void main(String args[]) {
		long startTime, stopTime;
		double ms;


		// Create random array
		int array[] = new int[SIZE];
		// Create response array
		int res[] = new int[SIZE];

		// Create threads array
		Example8 threads[] = new Example8[MAXTHREADS];

		ms = 0;
		System.out.println("Starting testing...");
		for (int j = 1; j <= Utils.N; j++) {
			Utils.randomArray(array);
			if (j == Utils.N) {
				Utils.displayArray("before", array);
			}

			// Initialize threads
			for (int i = 0; i < MAXTHREADS; i++) {
				threads[i] = new Example8(i, array, res);
			}
			startTime = System.currentTimeMillis();
			// Start threads
			for (int i = 0; i < MAXTHREADS; i++) {
				threads[i].start();
			}
			// Stop threads
			for (int i = 0; i < MAXTHREADS; i++) {
				try {
					threads[i].join();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			stopTime = System.currentTimeMillis();
			ms += (stopTime - startTime);
			
			// Get last result
			if (j == Utils.N) {
				Utils.displayArray("after", res);
			}
		}
		System.out.printf("avg time = %.5f ms\n", (ms / Utils.N));
	}
}
