// =================================================================
//
// File: Example7.java
// Author(s):
//					 A01651517 Pedro Luis Gonzalez Roa 
//					 A01703947 Juan Alejandro Alcantara Minaya
//
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

public class Example7 {
	private static final int SIZE = 1_000_000;
	private boolean array[];

	public Example7(boolean array[]) {
		this.array = array;
	}

	public void calculate() {
		for (int i = 2; i < array.length; i++) {
			array[i] = true;
			for (int j = 2; j <= Math.sqrt(i); j++) {
				if (i % j == 0) {
					array[i] = false;
					break;
				}
			}
		}
	}

	public static void main(String args[]) {
		boolean array[] = new boolean[SIZE + 1];
		long startTime, stopTime;
		double acum = 0;

		System.out.println("At first, neither is a prime. We will display to TOP_VALUE:");
		for (int i = 2; i < Utils.TOP_VALUE; i++) {
			array[i] = false;
			System.out.print("" + i + ", ");
		}
		System.out.println("");

		// Create the object here.
		Example7 example = new Example7(array);
		acum = 0;
		System.out.printf("Starting...\n");
		for (int i = 0; i < Utils.N; i++) {
			startTime = System.currentTimeMillis();

			// Call yout method here.
			example.calculate();

			stopTime = System.currentTimeMillis();

			acum += (stopTime - startTime);
		}
		System.out.println("Expanding the numbers that are prime to TOP_VALUE:");
		for (int i = 2; i < Utils.TOP_VALUE; i++) {
			if (array[i]) {
				System.out.print("" + i + ", ");
			}
		}
		System.out.println("");
		System.out.printf("avg time = %.5f ms\n", (acum / Utils.N));
	}
}
