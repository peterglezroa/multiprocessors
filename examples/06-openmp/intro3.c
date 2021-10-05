#include <stdio.h>
#include <omp.h>

int main(int argc, char* argv[]) {
	printf("Using the share clause\n");
	int x = 1;
	#pragma omp parallel shared(x) num_threads(3)
	{
		x++;
		printf("In the parallel block, x is %i\n", x);
	}
	printf("Outside the parallel block, x is %i\n", x);

	printf("\nUsing the private clause\n");
	x = 2;
	#pragma omp parallel private(x) num_threads(4)
	{
		x++;
		printf("In the parallel block, x is %i\n", x);
	}
	printf("Outside the parallel block, x is %i\n", x);

	printf("\nUsing the firstprivate clause\n");
	x = 2;
	#pragma omp parallel firstprivate(x) num_threads(2)
	{
		x++;
		printf("In the parallel block, x is %i\n", x);
	}
	printf("Outside the parallel block, x is %i\n", x);

	return 0;
}
