#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>

#define ITERATIONS 10

struct timeval startTime, stopTime;
int started = 0;

// =================================================================
// Records the initial execution time.
// @author: Manchas2k4
// =================================================================
void start_timer() {
	started = 1;
	gettimeofday(&startTime, NULL);
}

// =================================================================
// Calculates the number of microseconds that have elapsed since
// the initial time.
//
// @returns the time passed
// @author: Manchas2k4
// =================================================================
double stop_timer() {
	long seconds, useconds;
	double duration = -1;

	if (started) {
		gettimeofday(&stopTime, NULL);
		seconds  = stopTime.tv_sec  - startTime.tv_sec;
		useconds = stopTime.tv_usec - startTime.tv_usec;
		duration = (seconds * 1000.0) + (useconds / 1000.0);
		started = 0;
	}
	return duration;
}

// =================================================================
// Class that contains all the relevant data and the method to extract
// said data for the implementations of Convolution.
// =================================================================
class ConvContext {
    private:
        cv::Mat src, dst;
        int fRows, fCols;
        float *filter;
        bool debug;

        /* Function to print if debugging */
        void log(const char str[]) {
            if (log != NULL) fprintf(log, "%s\n", str);
        }

        /* Scan filter using scanf */
        float * scanFilter(int fRows, int fCols) {
            float *filter;

            // Make space in memory
            filter = (float *)malloc(sizeof(float)*fRows*fCols);

            // Scan filter
            if (log != NULL) printf(log, "Give me the filter: \n");
            for (int i = 0; i < fRows*fCols; i++) fscanf(stdin, "%f", &filter[i]);

            return filter;
        }

    public:
        ConvContext(std::string imagePath, bool color, FILE *log = NULL)
        : debug(debug) {
            // Read image
            debugPrintf("Reading image....");
            src = (color)?
                cv::imread(imagePath, cv::IMREAD_COLOR):
                cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

            // Scan filter
            debugPrintf("Give me filter dimensions (rows cols): ");
            fscanf(stdin, "%i %i", &fRows, &fCols);
            filter = scanFilter(fRows, fCols);
            if (fCols <= 0 || fRows <= 0)
                throw std::runtime_error("Invalid filter dimensions!");
        }

        ~ConvContext() { free(filter); }

        void setDestination(uchar *dstRaw) {
            dst = cv::Mat(
                src.rows,
                src.cols,
                src.type(),
                dstRaw,
                cv::Mat::AUTO_STEP
            );
            free(dstRaw);
        }

        /* Function to display the original image and the destination image */
        void display() {
            cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
            cv::imshow("Original", src);

            cv::namedWindow("Convolved", cv::WINDOW_AUTOSIZE);
            cv::imshow("Convolved", dst);

            cv::waitKey(0);
        }

        // Getters
        const cv::Mat *getSrc() const { return &src; }
        const cv::Mat *getDst() const { return &dst; }
        int getFRows() { return fRows; }
        int getFCols() { return fCols; }
        int getFSize() { return fRows*fCols; }
        float *getFilter() const { return filter; }
        int getRows() { return src.rows; }
        int getCols() { return src.cols; }
        int getChannels() { return src.channels(); }
        int imgSize() { return src.cols*src.rows; }
        int getSize() { return src.rows*src.cols*src.channels; }
};

#endif
