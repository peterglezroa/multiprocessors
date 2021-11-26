/*----------------------------------------------------------------
* Programación avanzada: Proyecto final
* Fecha: 25-Nov-2021
* Autor: A01651517 Pedro González
*--------------------------------------------------------------*/
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;
import java.io.IOException;
import java.util.Scanner;

public class Convolution {
    public static final int ITERATIONS = 100;
	private int src[], dst[], rows, cols;
    private float kernel[];
    private int kRows, kCols;

	public Convolution(int src[], int dst[], int rows, int cols,
    float kernel[], int kRows, int kCols) {
		this.src = src; this.dst = dst;
		this.cols = cols; this.rows = rows;

        this.kernel = kernel; this.kRows = kRows; this.kCols = kCols;
	}

	private void convolution() {
        int size = rows*cols, ksize = kRows*kCols;
        int spos, pos, pixel, dpixel;
        float r, g, b;

        for (int i = 0; i < size; i++) {
            r = 0; g = 0; b = 0;
            spos = i - (int)Math.floor(ksize/2);
            for (int k = 0; k < ksize; k++) {
                pos = spos + k;
                if (pos > 0 && pos < size) {
                    pixel = src[spos+k];
                    r += (float)((pixel & 0x00ff0000) >> 16) * kernel[k];
                    g += (float)((pixel & 0x0000ff00) >> 8) * kernel[k];
                    b += (float)((pixel & 0x000000ff) >> 0) * kernel[k];
                }
            }

            dpixel = (0xff000000)
				| (((int)r) << 16)
				| (((int)g) << 8)
				| (((int)b) << 0);
            dst[i] = dpixel;
        }
	}

	public static void main(String args[]) throws Exception {
		long startTime, stopTime;
		double ms;

		if (args.length != 1) {
			System.out.println("usage: java Example10 image_file");
			System.exit(-1);
		}

        // Read image
        final BufferedImage source = ImageIO.read(new File(args[0]));

		int rows = source.getHeight();
		int cols = source.getWidth();
		int src[] = source.getRGB(0, 0, cols, rows, null, 0, cols);
		int dst[] = new int[src.length];

        // Scan kernel
        Scanner scanner = new Scanner(System.in);
        int kRows = scanner.nextInt();
        int kCols = scanner.nextInt();
        float kernel[] = new float[kRows*kCols];

        for (int i = 0; i < kRows*kCols; i++) kernel[i] = scanner.nextFloat();

		ms = 0;
		Convolution conv = new Convolution(src, dst, rows, cols, kernel, kRows,
            kCols);
		for (int i = 0; i < ITERATIONS; i++) {
			startTime = System.currentTimeMillis();

			conv.convolution();

			stopTime = System.currentTimeMillis();
			ms += (stopTime - startTime);
		}
		System.out.printf("avg time = %.4f\n", ms/ITERATIONS);
		final BufferedImage destination = new BufferedImage(cols, rows,
            BufferedImage.TYPE_INT_ARGB);
		destination.setRGB(0, 0, cols, rows, dst, 0, cols);

		try {
			ImageIO.write(destination, "png", new File("conv.png"));
			System.out.println("Image was written succesfully.");
		} catch (IOException ioe) {
			System.out.println("Exception occured :" + ioe.getMessage());
		}
	}
}
