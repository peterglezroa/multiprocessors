// =================================================================
//
// File: Example8.java
// Authors:
//				 A01651517 Peter Glez
//				 A01703947 Juan Alcántara
// Description: This file implements the enumeration sort algorithm
// 				using Java's Fork-Join.
//
// Copyright (c) 2021 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ForkJoinPool;

public class Example8 extends RecursiveAction {
	private static final int SIZE = 100_000;
	private static final int MIN = 1_000;

	private int array[], temp[], start, end;
	
  // place your code here
	public Example8 (int array[], int temp[], int start, int end) {
		this.array = array;
		this.temp = temp;
		this.start = start;
		this.end = end;
	}

	int count;
	private void computeDirectly() {
		for (int i = start; i < end; i++) {
			count = 0;
			for (int j = start; j < array.length; j++) {
				if (
						array[i] > array[j]
						|| array[i] == array[j] && i < j
				) {
					count++;
				}
			}
			temp[count] = array[i];
		}
	}

	@Override
	protected void compute() {
		if ((start - end) <= MIN) {
			computeDirectly();
		} else {
			int mid = (start + end)/2;
			invokeAll(
					new Example8(array, temp, start, mid),
					new Example8(array, temp, mid, end)
			);
		}
	}

	public static void main (String args[]) {
		long startTime, stopTime;
		int array[] = new int[SIZE];
		int temp[] = new int[SIZE];
		double ms;
		ForkJoinPool pool;

		Utils.randomArray(array);
		Utils.displayArray("before", array);

		// Fill with 0s the temp array
		for (int i = 0; i < temp.length; i++) {
			temp[i] = 0;
		}

		System.out.printf("Starting with %d threads...\n", Utils.MAXTHREADS);
		ms = 0;
		for (int i = 1; i <= Utils.N; i++) {
			pool = new ForkJoinPool(Utils.MAXTHREADS);
			startTime = System.currentTimeMillis();
			pool.invoke(new Example8(array, temp, 2, array.length));

			stopTime = System.currentTimeMillis();
			ms += (stopTime - startTime);
		}
		Utils.displayArray("after", temp);
		System.out.printf("\navg time = %.5f ms\n", (ms / Utils.N));
	}
}
