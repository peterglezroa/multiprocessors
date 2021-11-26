#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "utils.h"

#define TPB 256

__global__
void convolution(const uchar *src, const int channels, const float *kernel,
const int kSize, uchar *dst, const int size) {
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    if (tid < size) {
        int spos = tid - (int)(kSize/2)*channels;
        uchar byte = 0;
        for (int i = 0; i < kSize; i++)
            byte += src[spos+i*channels]*kernel[i];
        dst[tid] = byte;
    }
}

int main(int argc, char *argv[]) {
    ConvContext *context;
    double ms;
    bool grayscale = false;
    uchar *gpu_src, *gpu_dst, *dstRaw;
    float *gpu_kernel;

    if (argc == 3 && strcmp(argv[2], "--gray") == 0) grayscale = true;
    else if (argc != 2) {
        fprintf(stderr, "usage: %s source_file\n", argv[0]);
        return -1;
    }

    // Get context
    context = new ConvContext(argv[1], grayscale);
    context->printSize(stdout);

    // Upload image and kernel to gpu
    fprintf(stdout, "Uploading image and kernel to GPU...\n");
    cudaMalloc((void**) &gpu_src, sizeof(uchar)*context->getSize());
    cudaMemcpy(gpu_src, context->getData(), sizeof(uchar)*context->getSize(),
        cudaMemcpyHostToDevice);
    cudaMalloc((void**) &gpu_kernel, sizeof(float)*context->getKSize());
    cudaMemcpy(gpu_kernel, context->getKernel(), sizeof(float)*context->getKSize(),
        cudaMemcpyHostToDevice);

    // Make space for result
    dstRaw = (uchar *)malloc(sizeof(uchar) * context->getSize());
    cudaMalloc((void**) &gpu_dst, sizeof(uchar) * context->getSize());

    // Run algorithm n times
    for (int i = 0; i < ITERATIONS; i++) {
        start_timer();

        convolution<<<context->getSize()/TPB, TPB>>>(gpu_src,
        context->getChannels(), gpu_kernel, context->getKSize(), gpu_dst,
        context->getSize());

        ms += stop_timer()/ITERATIONS;
    }
    fprintf(stdout, "Calculation time: %.5f ms\n", ms);

    // Copy processed image to CPU
    cudaMemcpy(dstRaw, gpu_dst, sizeof(uchar)*context->getSize(),
        cudaMemcpyDeviceToHost);
    context->setDestination(dstRaw);
//    context->display();

    cudaFree(gpu_src); cudaFree(gpu_dst); cudaFree(gpu_kernel);
    free(dstRaw);
    return 0;
}
