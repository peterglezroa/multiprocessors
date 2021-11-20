#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define TPB 256

// Stride jump = {tid%[(cols - Fcols)/stride + 1]} * (cols+1)
__global__
void convolution(const uchar *src, const int cols, const float *filter,
const int fSize, const int stride, uchar *dst, const int dCols, const int size) {
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    int spos = stride*((tid%dCols)+(int)(tid/dCols*cols));
    uchar byte;
    if (tid < size) {
        byte = 0;
        for (int i = 0; i < fSize; i++)
            byte += src[spos+i]*filter[i];
        dst[tid] = byte;
    }
}

int main(int argc, char *argv[]) {
    cv::Mat src, dst;
    int size, fRows, fCols, stride;
    uchar *gpu_src, *gpu_dst, *dstRaw;
    float *filter, *gpu_filter;

    if (argc != 2) {
        fprintf(stderr, "usage: %s source_file\n", argv[0]);
        return -1;
    }

    // Scan filter dimensions
    fprintf(stdout, "Give me filter dimensions (rows cols): ");
    fscanf(stdin, "%i %i", &fRows, &fCols);
    filter = (float *)malloc(sizeof(float)*fRows*fCols);

    // Scan filter
    fprintf(stdout, "Give me the filter: \n");
    for (int i = 0; i < fRows*fCols; i++) fscanf(stdin, "%f", &filter[i]);

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
    fprintf(stdout, "Image: %s => rows: %i, cols: %i, channels: %i\n",
        argv[1], src.rows, src.cols, src.channels());

    // Upload image and filter to gpu
    fprintf(stdout, "Uploading image and filter to GPU...\n");
    cudaMalloc((void**) &gpu_filter, sizeof(float)*fRows*fCols);
    cudaMalloc((void**) &gpu_src, sizeof(uchar) * size * src.channels());
    cudaMemcpy(gpu_filter, filter, sizeof(float) * fRows * fCols,
        cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_src, src.data, sizeof(uchar) * size * src.channels(),
        cudaMemcpyHostToDevice);

    // Calculate sizes
    int dRows = (src.rows-fRows)/stride + 1;
    int dCols = (src.cols-fCols)/stride + 1;
    int dSize = dRows*dCols;

    dstRaw = (uchar *)malloc(sizeof(uchar) * dSize);
    cudaMalloc((void**) &gpu_dst, sizeof(uchar) * dSize);

    convolution<<<dSize/TPB + 1, TPB>>>(gpu_src, src.cols, gpu_filter, fRows*fCols,
        stride, gpu_dst, dCols, dSize);

    // Copy processed image to CPU
    cudaMemcpy(dstRaw, gpu_dst, sizeof(uchar) * dSize, cudaMemcpyDeviceToHost);

    // Convert result to opencv
    dst = cv::Mat(dRows, dCols, CV_8UC1, dstRaw, cv::Mat::AUTO_STEP);

    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original", src);

    cv::namedWindow("GrayScale", cv::WINDOW_AUTOSIZE);
    cv::imshow("GrayScale", dst);

    cv::waitKey(0);

    cudaFree(gpu_src); cudaFree(gpu_dst);
    free(filter); free(dstRaw);
    return 0;
}
