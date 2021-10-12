// =================================================================
//
// File: Example7.java
// Authors:
//				 A01651517 Peter Glez
//				 A01703947 Juan Alcántara
// Description: This file contains the code to brute-force all
//				prime numbers less than MAXIMUM using Java's
//				Fork-Join.
//
// Copyright (c) 2021 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.util.Arrays;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ForkJoinPool;

public class Example7 extends RecursiveAction {
	private static final int SIZE = 1_000_000;
	private static final int MIN = 1_000;
	private int start, end;
	private boolean array[];

	public Example7 (boolean array[], int start, int end) {
		this.array = array;
		this.start = start;
		this.end = end;
	}

	void computeDirectly() {
		for (int i = start; i < end; i++) {
			array[i] = true;
			for (int j = 2; j <= Math.sqrt(i); j++) {
				if (i % j == 0) {
					array[i] = false;
					break;
				}
			}
		}
	}

	@Override
	protected void compute() {
		if ((start - end) <= MIN) {
			computeDirectly();
		} else {
			int mid = (start + end)/2;
			invokeAll(
					new Example7(array, start, mid),
					new Example7(array, mid, end) 
			);
		}
	}

	public static void main(String args[]) {
		long startTime, stopTime;
		boolean array[];
		double ms;
		ForkJoinPool pool;

		array = new boolean[SIZE];

		System.out.printf("Starting with %d threads...\n", Utils.MAXTHREADS);
		ms = 0;
		for (int i = 1; i <= Utils.N; i++) {

			pool = new ForkJoinPool(Utils.MAXTHREADS);

			startTime = System.currentTimeMillis();

			pool.invoke(new Example7(array, 2, array.length));

			stopTime = System.currentTimeMillis();
			ms += (stopTime - startTime);
			if (i == Utils.N) 
				for (int j = 2; j < 100; j++)
					if (array[j])
						System.out.printf("%d ", j);
		}
		System.out.printf("\navg time = %.5f ms\n", (ms / Utils.N));
	}

}
