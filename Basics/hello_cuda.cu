#include <stdio.h>
#include <stdlib.h>

__global__ void devicePrint(void) {
	printf("Hello from GPU -- Thread [%d, %d]\n", threadIdx.x, blockIdx.x);
}

int main(int argc, char** argv) {
	printf("Hello World from Host Device\n");
	devicePrint<<<1,1>>>();
	cudaDeviceSynchronize();
	return 0;
}
