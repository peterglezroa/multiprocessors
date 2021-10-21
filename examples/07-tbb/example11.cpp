// =================================================================
//
// File: example10.cpp
// Author(s):
// Description: This file implements the code that transforms a
//				grayscale image using Intel's TBB. Uses OpenCV, to compile:
//			  g++ example10.cpp `pkg-config --cflags --libs opencv4` -ltbb
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include "utils.h"

using namespace std;
using namespace tbb;
using namespace cv;

class GrayoMcQueen {
	private:
    cv::Mat *src, *dest;

	public:
		GrayoMcQueen(cv::Mat *src, cv::Mat *dest) : src(src), dest(dest) {}

    void operator() (const blocked_range<int> &r) const {
      for (int i = r.begin(); i < r.end(); i++) {
        float calc = (
            (float) src->at<cv::Vec3b>(i)[RED]+
            (float) src->at<cv::Vec3b>(i)[GREEN]+
            (float) src->at<cv::Vec3b>(i)[BLUE]
        )/3;

        dest->at<cv::Vec3b>(i)[RED] = calc;
        dest->at<cv::Vec3b>(i)[GREEN] = calc;
        dest->at<cv::Vec3b>(i)[BLUE] = calc;
      }
    }
};

int main(int argc, char* argv[]) {
	int i;
	double acum;

	if (argc != 2) {
	printf("usage: %s source_file\n", argv[0]);
		return -1;
	}

	cv::Mat src = cv::imread(argv[1], cv::IMREAD_COLOR);
	cv::Mat dest = cv::Mat(src.rows, src.cols, CV_8UC3);
	if (!src.data) {
    printf("Could not load image file: %s\n", argv[1]);
		return -1;
	}

	acum = 0;
	for (i = 0; i < N; i++) {
    GrayoMcQueen cuchao(&src, &dest);
		start_timer();

    parallel_for(blocked_range<int>(0, src.cols*src.rows), cuchao);

		acum += stop_timer();
	}

	printf("avg time = %.5lf ms\n", (acum / N));

	cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original", src);

	cv::namedWindow("Gray", cv::WINDOW_AUTOSIZE);
    cv::imshow("Gray", dest);

	cv::waitKey(0);
	cv::imwrite("gray_scale.png", dest);

	return 0;
}
