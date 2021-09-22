// =================================================================
//
// File: Example7.java
// Author(s): Peter Glez A01651517 && Juan Alcantara A01703947
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
    private int start, finish;
    
    public Example7(boolean array[], int start, int finish) {
        this.array = array;
        this.start = start;
        this.finish = finish;
    }

    public boolean[] getArray() { return array; }

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
        boolean array[] = new boolean[MAXIMUM + 1];
        int blocks;
        long startTime, stopTime;
        Example7 threads[];
        double acum = 0;

        // Calculate how many threads we can use
        blocks = MAXIMUM / Utils.MAXTHREADS;
        threads = new Example7[Utils.MAXTHREADS];

        acum = 0;
        System.out.printf("Starting with %d threads...\n", Utils.MAXTHREADS);
        for (int j = 0; j <= Utils.N; j++) {

            // Create threads
            for (int i = 0; i < threads.length; i++) {
                if (i == 0) {
                    threads[i] = new Example7(array, 2, ((i+1) * blocks));
                } else if (i != threads.length-1) {
                    threads[i] = new Example7(array, (i*blocks), ((i+1) * blocks));
                } else {
                    threads[i] = new Example7(array, (i*blocks), MAXIMUM);
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
                for (Example7 thread : threads) {
                    for (int i = thread.start; i < thread.finish; i++)
                        if (thread.getArray()[i])
                            System.out.printf("%d ", i);
                }
            }
        }
        System.out.printf("\navg time = %.5f ms\n", (acum / Utils.N));
    }
}
