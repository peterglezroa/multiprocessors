#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define TPB 256

__global__
void grayScaleImage(const uchar *src, uchar *dst, const int size,
const int channels) {
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    uchar byte;

    if (tid < size) {
        byte = 0;
        for (int j = 0; j < channels; j++) byte += src[tid*channels + j]/3;
        dst[tid] = byte;
    }
}

int main(int argc, char* argv[]) {
    cv::Mat src, dst;
    int size;
    uchar *gpu_src, *gpu_dst, *dstRaw;

    if (argc != 2) {
        fprintf(stderr, "usage: %s source_file\n", argv[0]);
        return -1;
    }

    fprintf(stdout, "Reading image...\n");
    src = cv::imread(argv[1], cv::IMREAD_COLOR);
    size = src.rows*src.cols;
    fprintf(stdout, "Image: %s => rows: %i, cols: %i, channels: %i\n",
        argv[1], src.rows, src.cols, src.channels());

    // Make space for raw dst
    dstRaw = (uchar *)malloc(sizeof(uchar) * size);

    // Upload image to gpu
    fprintf(stdout, "Uploading image to GPU...\n");
    cudaMalloc((void**) &gpu_src, sizeof(uchar) * size * src.channels());
    cudaMalloc((void**) &gpu_dst, sizeof(uchar) * size);
    cudaMemcpy(gpu_src, src.data, sizeof(uchar) * size * src.channels(),
    cudaMemcpyHostToDevice);

    grayScaleImage<<<(size/TPB)+1, TPB>>>(gpu_src,gpu_dst,size,src.channels());

    // Copy processed image to CPU
    fprintf(stdout, "Done!\n");
    cudaMemcpy(dstRaw, gpu_dst, sizeof(uchar) * size, cudaMemcpyDeviceToHost);
    dst = cv::Mat(src.rows, src.cols, CV_8UC1, dstRaw, cv::Mat::AUTO_STEP);

    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original", src);

    cv::namedWindow("GrayScale", cv::WINDOW_AUTOSIZE);
    cv::imshow("GrayScale", dst);

    cv::waitKey(0);
    return 0;
}
