// =================================================================
//
// File: Example4.java
// Authors:
// Description: This file contains the code to count the number of
//                even numbers within an array using Threads.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

public class Example4 extends Thread {
    private static final int SIZE = 100_000_000;
    private int array[];
    private int result;
    private int start; // ptr array start
    private int finish; // ptr array finish
    
    public Example4(int array[], int start, int finish) {
        this.array = array;
        this.result = 0;
        this.start = start;
        this.finish = finish;
    }

    public int getResult() { return result; }

    public void run() {
        for (int i = start; i < finish; i++) {
            if (array[i]%2 == 0)
                result++;
        }
    }

    public static void main(String args[]) {
        int array[] = new int[SIZE], blocks, res = 0;
        long startTime, stopTime;
        Example4 threads[];
        double acum = 0;

        Utils.fillArray(array);
        Utils.displayArray("array", array);

        // Calculate how many threads we can use
        blocks = SIZE / Utils.MAXTHREADS;
        threads = new Example4[Utils.MAXTHREADS];

        acum = 0;
        System.out.printf("Starting with %d threads...\n", Utils.MAXTHREADS);
        for (int j = 0; j <= Utils.N; j++) {

            // Create threads
            for (int i = 0; i < threads.length; i++) {
                if (i != threads.length-1) {
                    threads[i] = new Example4(array, (i*blocks), ((i+1) * blocks));
                } else {
                    threads[i] = new Example4(array, (i*blocks), SIZE);
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
