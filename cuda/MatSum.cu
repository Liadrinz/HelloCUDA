// 矩阵加法
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <time.h>

using namespace std;

__global__ void Plus(float A[], float B[], float C[], int n) {
	// GPU上的代码
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	C[i] = A[i] + B[i];
}

int main() {
	float *A, *Ad, *B, *Bd, *C, *Cd;
	int n = 1024 * 1024;
	int size = n * sizeof(float);

	// CPU分配内存
	A = (float*)malloc(size);
	B = (float*)malloc(size);
	C = (float*)malloc(size);

	for (int i = 0; i < n; i++) {
		A[i] = 90.0;
		B[i] = 10.0;
	}

	// GPU分配内存
	cudaMalloc((void**)&Ad, size);
	cudaMalloc((void**)&Bd, size);
	cudaMalloc((void**)&Cd, size);

	// 将CPU数据拷贝到GPU
	cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);
	cudaMemcpy(Cd, C, size, cudaMemcpyHostToDevice);

	dim3 blockNum(n / 512);
	dim3 threadPerBlock(512);
	Plus<<<blockNum, threadPerBlock>>>(Ad, Bd, Cd, n);
	
	cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);

	float max_error = 0.0;
	for (int i = 0; i < n; i++)
	{
		max_error += fabs(100.0 - C[i]);
	}

	cout << "max error is " << max_error << endl;
	
	free(A);
	free(B);
	free(C);
	cudaFree(Ad);
	cudaFree(Bd);
	cudaFree(Cd);
	return 0;
}