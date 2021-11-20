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
void convolution(const uchar *src, const int channels, const float *kernel,
const int kSize, const int stride, uchar *dst, const int size) {
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    if (tid < size) {
        int spos = tid - (int)(kSize/2)*channels;
        uchar byte = 0;
        for (int i = 0; i < kSize; i++) {
            byte += src[spos+i*channels]*kernel[i];
        }
        dst[tid] = byte;
    }
}

int main(int argc, char *argv[]) {
    bool gray = false;
    cv::Mat src, dst;
    int size, kRows, kCols, stride;
    uchar *gpu_src, *gpu_dst, *dstRaw;
    float *kernel, *gpu_kernel;

    if (argc > 2 && strcmp(argv[2], "--grey") == 0) gray = true;
    else if (argc != 2) {
        fprintf(stderr, "usage: %s source_file\n", argv[0]);
        return -1;
    }

    // Scan kernel dimensions
    fprintf(stdout, "Give me kernel dimensions (rows cols): ");
    fscanf(stdin, "%i %i", &kRows, &kCols);
    kernel = (float *)malloc(sizeof(float)*kRows*kCols);

    // Scan kernel 
    fprintf(stdout, "Give me the kernel: \n");
    for (int i = 0; i < kRows*kCols; i++) fscanf(stdin, "%f", &kernel[i]);

    fprintf(stdout, "Give me a stride: ");
    fscanf(stdin, "%i", &stride);
    fprintf(stdout, "\n");

    if (stride > kCols || stride > kRows) {
        fprintf(stderr, "Stride cannot be bigger than the dimensions of the kernel");
        return -2;
    }

    // Read image
    fprintf(stdout, "Reading image...\n");
    src = (!gray)?
        cv::imread(argv[1], cv::IMREAD_COLOR):
        cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    size = src.rows*src.cols;
    fprintf(stdout, "Image: %s => rows: %i, cols: %i, channels: %i\n",
        argv[1], src.rows, src.cols, src.channels());

    // Upload image and kernel to gpu
    fprintf(stdout, "Uploading image and kernel to GPU...\n");
    cudaMalloc((void**) &gpu_kernel, sizeof(float)*kRows*kCols);
    cudaMalloc((void**) &gpu_src, sizeof(uchar)*size*src.channels());
    cudaMemcpy(gpu_kernel, kernel, sizeof(float)*kRows*kCols,
        cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_src, src.data, sizeof(uchar)*size*src.channels(),
        cudaMemcpyHostToDevice);

    // Calculate sizes
    int dRows = src.rows/stride;
    int dCols = src.cols/stride;
    int dSize = dRows*dCols*src.channels();

    fprintf(stdout, "New Image => rows: %i, cols: %i, channels: %i\n",
        dRows, dCols, src.channels());

    dstRaw = (uchar *)malloc(sizeof(uchar) * dSize);
    cudaMalloc((void**) &gpu_dst, sizeof(uchar) * dSize);

    convolution<<<dSize/TPB + 1, TPB>>>(gpu_src, src.channels(), gpu_kernel,
        kRows*kCols, stride, gpu_dst, dSize);

    // Copy processed image to CPU
    cudaMemcpy(dstRaw, gpu_dst, sizeof(uchar) * dSize, cudaMemcpyDeviceToHost);

    // Convert result to opencv
    dst = cv::Mat(dRows, dCols, src.type(), dstRaw, cv::Mat::AUTO_STEP);

    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original", src);

    cv::namedWindow("GrayScale", cv::WINDOW_AUTOSIZE);
    cv::imshow("GrayScale", dst);

    cv::waitKey(0);

    cudaFree(gpu_src); cudaFree(gpu_dst); cudaFree(gpu_kernel);
    free(kernel); free(dstRaw);
    return 0;
}
