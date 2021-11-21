#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "utils.h"

uchar * convolution(ConvContext *context) {
    // Make space for destination
    uchar *dst;
    dst = (uchar *)malloc(sizeof(uchar) * context->getSize());
    return dst;
}

int main(int argc, char *argv[]) {
    ConvContext *context;
    int ms;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <image file>\n", argv[0]);
        return -1;
    }

    // Get context
    context = new ConvContext(argv[1], false, true);

    // Run algorithm n times
    for (int i = 0; i < ITERATIONS; i++) {
        start_timer();

        // Only save on last iteration
        if (i == ITERATIONS-1) context->setDestination(convolution(context));
        else convolution(context);

        ms += stop_timer()/ITERATIONS;
    }

//    context.show();

    delete context;
    free(context);
    return 0;
}
