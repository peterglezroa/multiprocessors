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
        int kRows, kCols;
        float *kernel;

        /* Scan kernel using scanf */
        float * scanKernel() {
            float *kernel;

            // Make space in memory
            kernel = (float *)malloc(sizeof(float)*kRows*kCols);

            // Scan kernel 
            for (int i = 0; i < kRows*kCols; i++) fscanf(stdin, "%f", &kernel[i]);

            return kernel;
        }

    public:
        ConvContext(std::string imagePath, bool color) {
            // Read image
            src = (color)?
                cv::imread(imagePath, cv::IMREAD_COLOR):
                cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

            // Scan kernel 
            fscanf(stdin, "%i %i", &kRows, &kCols);
            if (kCols <= 0 || kRows <= 0)
                throw std::runtime_error("Invalid kernel dimensions!");
            kernel = scanKernel();
        }

        ~ConvContext() { free(kernel); }

        void setDestination(uchar *dstRaw) {
            dst = cv::Mat(
                src.rows,
                src.cols,
                src.type(),
                dstRaw,
                cv::Mat::AUTO_STEP
            );
        }

        /* Function to display the original image and the destination image */
        void display() {
            cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
            cv::imshow("Original", src);

            cv::namedWindow("Convolved", cv::WINDOW_AUTOSIZE);
            cv::imshow("Convolved", dst);

            cv::waitKey(0);
        }

        void printSize(FILE *file) {
            fprintf(file, "Image size: %ix%ix%i = %i\n",
                src.rows, src.cols, src.channels(), getSize());
        }

        // Getters
        const cv::Mat *getSrc() { return &src; }
        const cv::Mat *getDst() { return &dst; }
        uchar *getData() const { return src.data; }
        float *getKernel() const { return kernel; }
        int getKRows() { return kRows; }
        int getKCols() { return kCols; }
        int getKSize() { return kRows*kCols; }
        int getRows() { return src.rows; }
        int getCols() { return src.cols; }
        int getChannels() { return src.channels(); }
        int imgSize() { return src.cols*src.rows; }
        int getSize() { return src.rows*src.cols*src.channels(); }
};

#endif
