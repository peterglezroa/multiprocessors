// =================================================================
//
// File: Example11.java
// Author(s):
//				 A01651517 Peter Glez
//				 A01703947 Juan Alcántara
// Description: This file implements the code that transforms a
//				grayscale image using Java's Fork-Join.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;
import java.io.IOException;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ForkJoinPool;

public class Example11 extends RecursiveAction {
    private static final int MIN = 15_000;
	private int src[], dest[], start, end, width, height;

    public Example11(int src[], int dest[], int width, int height, int start, int end) {
        this.src = src;
        this.dest = dest;
        this.start = start;
        this.end = end;
    }

    private void grayScale(int index) {
        int red, green, blue, avg;
        red = (src[index]&0x00FF0000)>>16;
        green = (src[index]&0x0000FF00)>>8;
        blue = (src[index]&0x000000FF);
        avg = (red + green + blue)/3;
        dest[index] = 0xFF000000 | (avg << 16) | (avg << 8) | avg;
    }

    protected void computeDirectly() {
        int index, ren, col;
        for (index = start; index < end; index++)
            grayScale(index);
    }

    @Override
    protected void compute() {
        if ((start - end) <= MIN) {
            computeDirectly();
        } else {
            int mid = start + ((end - start) / 2);
            invokeAll(new Example11(src, dest, width, height, start, mid),
                new Example11(src, dest, width, height, mid, end));
        }
    }

    public static void main(String args[]) throws Exception {
        long startTime, stopTime;
        double ms;
        int src[], dest[], w, h;
        ForkJoinPool pool;

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
        dest = new int[src.length];

        System.out.printf("Starting with %d threads...\n", Utils.MAXTHREADS);
        ms = 0;
        for (int i = 0; i < Utils.N; i++) {
            startTime = System.currentTimeMillis();
            pool = new ForkJoinPool(Utils.MAXTHREADS);
            pool.invoke(new Example11(src, dest, w, h, 0, (w*h)));

            stopTime = System.currentTimeMillis();
            ms += (stopTime - startTime);
        }

        System.out.printf("avg time = %.5f\n", (ms / Utils.N));
		final BufferedImage destination = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
		destination.setRGB(0, 0, w, h, dest, 0, w);

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

		try {
			ImageIO.write(destination, "png", new File("gray.jpg"));
			System.out.println("Image was written succesfully.");
		} catch (IOException ioe) {
			System.out.println("Exception occured :" + ioe.getMessage());
		}
    }
}
