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

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include "utils.h"

using namespace tbb;

class ConvolutionTBB {
private:
    ConvContext *context;
    uchar *dst;

public:
    ConvolutionTBB(ConvContext *context_) : context(context_) {
        dst = (uchar *)malloc(sizeof(uchar) * context->getSize());
    }

    void operator() (const blocked_range<int> &r) const {
        uchar byte;
        int spos, pos;

        for (int i = r.begin(); i != r.end(); i++) {
            byte = 0;
            spos = i - (int)(context->getKSize()/2)*context->getChannels();
            for (int f = 0; f < context->getKSize(); f++) {
                int pos = spos+f*context->getChannels();
                if (pos > 0 && pos < context->getSize())
                    byte += context->getData()[pos]*context->getKernel()[f];
            }
            dst[i] = byte;
        }
    }

    uchar * getRes() { return dst; }
};

int main(int argc, char *argv[]) {
    ConvContext *context;
    double ms;
    bool grayscale = true;

    if (argc == 3 && strcmp(argv[2], "--grey") == 0) grayscale = true;
    else if (argc != 2) {
        fprintf(stderr, "usage: %s source_file\n", argv[0]);
        return -1;
    }

    // Get context
    context = new ConvContext(argv[1], grayscale);
    context->printSize(stdout);

    // Run algorithm n times
    for (int i = 0; i < ITERATIONS; i++) {
        start_timer();

		ConvolutionTBB obj(context);
		parallel_for(blocked_range<int>(0, context->getSize()),  obj);

        // Only save on last iteration
        if (i == ITERATIONS-1) context->setDestination(obj.getRes());

        ms += stop_timer()/ITERATIONS;
        fprintf(stdout, "%.5f ms\n", ms);
    }
    fprintf(stdout, "Calculation time: %.5f ms\n", ms);

//    context->display();

    delete context;
    return 0;
}
