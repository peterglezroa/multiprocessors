// =================================================================
//
// File: Example7.java
// Author(s):
// Description: This file contains the code to brute-force all
//                prime numbers less than MAXIMUM. The time this
//                implementation takes will be used as the basis to
//                calculate the improvement obtained with parallel
//                technologies.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

public class Example7 extends Thread {
    private static final int MAXIMUM = 100_000_000;
    private boolean array[];
    
    public Example7(boolean array[], int start, int finish) {
        this.array = array;
        this.start = start;
        this.finish = finish;
    }

    public int getPrimes() { return primes; }

    public void run() {
        for (int i = start; i < finish; i++) {
            array[i] = true;
            for (int j = 2; j <= Math.sqrt(i); j++) {
                if (i%j == 0) {
                    array[i] = false;
                    break;
                }
            }
        }
    }

    public static void main(String args[]) {
        int array[] = new boolean[SIZE + 1], blocks;
        long startTime, stopTime;
        Example7 threads[];
        double acum = 0;

        // Calculate how many threads we can use
        blocks = SIZE / Utils.MAXTHREADS;
        threads = new Example7[Utils.MAXTHREADS];

        acum = 0;
        System.out.printf("Starting with %d threads...\n", Utils.MAXTHREADS);
        for (int j = 0; j <= Utils.N; j++) {

            // Create threads
            for (int i = 0; i < threads.length; i++) {
                if (i != threads.length-1) {
                    threads[i] = new Example7(array, (i*blocks), ((i+1) * blocks));
                } else {
                    threads[i] = new Example7(array, (i*blocks), SIZE);
                }
            }

            startTime = System.currentTimeMillis();

            // Run threads
            for (int i = 0; i < threads.length; i++) { threads[i].start(); }
            for (int i = 0; i < threads.length; i++) {
                try {
                    threads[i].join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

            stopTime = System.currentTimeMillis();
            acum += (stopTime - startTime);

            // Only get result of last run
            if (j == Utils.N) {
                for (Example4 thread : threads) { res += thread.result; }
            }
        }
        System.out.printf("sum = %d\n", res);
        System.out.printf("avg time = %.5f ms\n", (acum / Utils.N));
    }
}
