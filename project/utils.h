#ifndef UTILS_H
#define UTILS_H

#include <time.h>
#include <stdlib.h>
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
// Scans the filter from stdin.
//
// @returns pointer to filter information
// =================================================================
float * scanFilter(int fRows, int fCols) {
    float *filter;

    // Make space in memory
    filter = (float *)malloc(sizeof(float)*fRows*fCols);

    // Scan filter
    fprintf(stdout, "Give me the filter: \n");
    for (int i = 0; i < fRows*fCols; i++) fscanf(stdin, "%f", &filter[i]);

    return filter;
}
#endif
