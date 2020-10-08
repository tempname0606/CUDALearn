#include <stdlib.h>
#include <stdio.h>

#define N 512

__global__ void deviceAdd(int* a, int* b, int* c){
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void hostAdd(int* a, int* b, int* c){
	for(int index = 0; index < N; index++){
		c[index] = a[index] + b[index];
	}
}

void populateArray(int* data){
	for(int index = 0; index < N; index++){
		data[index] = index;
	}
}

void print(int* a, int* b, int* c){
	for(int index = 0; index < N; index++){
		printf("\n %d + %d = %d", a[index], b[index], c[index]);
	}
}

int main(int argc, char** argv){
	int *a, *b, *c;
	int *devA, *devB, *devC;
	int size = N * sizeof(int);

	// Allocate space for host copies of a, b, c
	a = (int*)malloc(size);
	b = (int*)malloc(size);
	c = (int*)malloc(size);

	populateArray(a);
	populateArray(b);

	// Allocate space for device copies of a, b, c
	// cudaMalloc ( void** devPtr, size_t size );
	cudaMalloc((void**)&devA, size);
	cudaMalloc((void**)&devB, size);
	cudaMalloc((void**)&devC, size);

	// cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
	cudaMemcpy(devA, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devB, b, N * sizeof(int), cudaMemcpyHostToDevice);
	
	// N blocks - 1 Thread per block
	deviceAdd<<<N, 1>>>(devA, devB, devC);
	
	// Copy results to host
	cudaMemcpy(c, devC, N * sizeof(int), cudaMemcpyDeviceToHost);

	
	print(a, b, c);
	
	free(a);
	free(b);
	free(c);

	// Free device memory
	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);

	return 0;
}
