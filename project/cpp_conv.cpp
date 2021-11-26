/*----------------------------------------------------------------
* Programación avanzada: Proyecto final
* Fecha: 25-Nov-2021
* Autor: A01651517 Pedro González
*--------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "utils.h"

uchar * convolution(ConvContext *context) {
    uchar *dst, byte, *dta = context->getData();
    float *k = context->getKernel();
    int spos, pos, channels = context->getChannels();

    // Make space for destination
    dst = (uchar *)malloc(sizeof(uchar) * context->getSize());

    for (int i = 0; i < context->getSize(); i++) {
        byte = 0;
        spos = i - (int)(context->getKSize()/2)*channels;
        for (int f = 0; f < context->getKSize(); f++) {
            int pos = spos+f*channels;
            if (pos > 0 && pos < context->getSize()) byte += dta[pos]*k[f];
        }
        dst[i] = byte;
    }

    return dst;
}

int main(int argc, char *argv[]) {
    ConvContext *context;
    double ms;
    uchar *dst;
    bool grayscale = true;

    if (argc != 2 && argc != 3) {
        fprintf(stderr, "usage: %s <image file>\n", argv[0]);
        return -1;
    }

    if (argc == 3) grayscale = false;

    // Get context
    context = new ConvContext(argv[1], grayscale);
    context->printSize(stdout);

    // Run algorithm n times
    for (int i = 0; i < ITERATIONS; i++) {
        start_timer();

        // Only save on last iteration
        if (i == ITERATIONS-1) dst = convolution(context);
        else convolution(context);

        ms += stop_timer()/ITERATIONS;
    }
    fprintf(stdout, "Calculation time: %.5f ms\n", ms);

    context->setDestination(dst);
    context->display();

    delete context;
    free(dst);
    return 0;
}
