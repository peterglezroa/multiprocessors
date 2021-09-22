// =================================================================
//
// File: Example11.java
// Author(s): Peter Glez A01651517,  Wuan Alcantara A01703947
// Description: This file implements the code that transforms a
//				grayscale image. The time this implementation takes will
//				be used as the basis to calculate the improvement obtained
// 				with parallel technologies.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

public class Example11 extends Thread {
    private int src[], dest[], start, finish;

    public Example11(int src[], int dest[], int start, int finish) {
        this.src = src;
        this.dest = dest;
        this.start = start;
        this.finish = finish;
    }

    public void run() {
        int red, green, blue, avg;
        for (int i = start; i < finish; i++) {
            red = (src[i] & 0x00FF0000) >> 16;
            green = (src[i] & 0x0000FF00) >> 8;
            avg = ((src[i]&0x00FF0000) + (src[i]&0x0000FF00) + (src[i]&0x000000FF))/3;
            dest[i] = 0xFF000000 | (avg << 16) | (avg << 8) | avg;
        }
    }

    public static void main(String args[]) throws Exception {
        long startTime, stopTime;
        double ms;
        Example11 threads[];
        int src[], dest[], w, h, blocks;

        if (args.length != 1) {
            System.out.println("usage: java Example11 image_file");
            System.exit(-1);
        }

        final String fileName = args[0];

        File srcFile = new File(fileName);
        final BufferedImage source = ImageIO.read(srcFile);

        w = source.getWidth();
        h = source.getHeight();

        src = source.getRGB(0, 0, w, h, null, 0, w);
        dest = new int [src.length];

        blocks = (w*h) / Utils.MAXTHREADS;
        threads = new Example11[Utils.MAXTHREADS];

        System.out.printf("Starting with %d threads...\n", Utils.MAXTHREADS);
        ms = 0;
        for (int j = 1; j <= Utils.N; j++) {
            for (int i = 0; i < threads.length; i++) {
                if (i != threads.length - 1) {
                    threads[i] = new Example11(src, dest, (i*blocks), ((i+1)*blocks));
                } else {
                    threads[i] = new Example11(src, dest, (i*blocks), (w*h));
                }
            }
            startTime = System.currentTimeMillis();
            for (Example11 thread : threads) { thread.start(); }
            for (Example11 thread : threads) {
                try {
                    thread.join();
                } catch (InterruptedException e) { e.printStackTrace();}
            }
            stopTime = System.currentTimeMillis();
            ms += (stopTime - startTime);
        }
        System.out.printf("avg time = %.5f\n", (ms/Utils.N));
        final BufferedImage destination = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        destination.setRGB(0, 0, w, h, dest, 0, w);

        /*
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            public void run() {
               ImageFrame.showImage("Original - " + fileName, source);
            }
        });

        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            public void run() {
               ImageFrame.showImage("Blur - " + fileName, destination);
            }
        });
        */
    }
}
