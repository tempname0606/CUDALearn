#include <stdio.h>
#include <stdlib.h>

__global__ void devicePrint(){
	// Print from GPU.
	printf("Hello from device! Thread %d,%d\n", threadIdx.x, blockIdx.x);
}

int main(int argc, char** argv){
	printf("Hello from host!\n");
	devicePrint<<<1, 1>>>();
	cudaDeviceSynchronize();
	return 0;
}
