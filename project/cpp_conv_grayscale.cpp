#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "utils.h"

int main(int argc, char *argv[]) {
    cv::Mat src, dst;
    int size, fRows, fCols, stride;
    float *filter;
    uchar *dstRaw;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <image file>\n", argv[0]);
        return -1;
    }

    // Scan filter dimensions and filter
    fprintf(stdout, "Give me filter dimensions (rows cols): ");
    fscanf(stdin, "%i %i", &fRows, &fCols);
    filter = scanFilter(fRows, fCols);

    fprintf(stdout, "Give me a stride: ");
    fscanf(stdin, "%i", &stride);
    fprintf(stdout, "\n");

    // Scan stride
    fprintf(stdout, "Give me a stride: ");
    fscanf(stdin, "%i", &stride);
    fprintf(stdout, "\n");
    if (stride > fCols || stride > fRows) {
        fprintf(stderr, "Stride cannot be bigger than the dimensions of the filter");
        return -2;
    }

    // Read image
    fprintf(stdout, "Reading image...\n");
    src = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    size = src.rows*src.cols;
    fprintf(stdout, "Image: %s => rows: %i, cols: %i, channels %i\n",
        argv[1], src.rows, src.cols, src.channels());

    // Calculate destination size
    int dRows = (src.rows-fRows)/stride + 1;
    int dCols = (src.cols-fCols)/stride + 1;
    int dSize = dRows*dCols;

    // Make space for destination
    dstRaw = (uchar *)malloc(sizeof(uchar)*dSize);

    // Run algorithm n times
    for (int i = 0; i < ITERATIONS; i++) {
        start_timer();
        // TODO: algorithm
        ms += stop_timer()/ITERATIONS;
    }

    dst = cv::Mat(dRows, dCols, CV_8UC1, dstRaw, cv::Mat::AUTO_STEP);
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    CV::imshow("Original", src);

    cv::namedWindow("Filtered", cv::WINDOW_AUTOSIZE);
    cv::imshow("Filtered", dst);

    cv::waitKey(0);

    free(filter); free(dstRaw);
    return 0;
}
