// 一维FFT
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../utils/Complex.cu"
#include <iostream>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <Windows.h>

int GetBits(int n) {
    int bits = 0;
    while (n >>= 1) {
        bits++;
    }
    return bits;
}

__device__ int br(int i, int bits) {
    int r = 0;
    do {
        r += i % 2 << --bits;
    } while (i /= 2);
    return r;
}

__device__ void Bufferfly(Complex *a, Complex *b, Complex factor) {
    Complex a1 = (*a) + factor * (*b);
    Complex b1 = (*a) - factor * (*b);
    *a = a1;
    *b = b1;
}

__global__ void FFT(Complex nums[], Complex result[], int n, int bits) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n) return;
    for (int i = 2; i < 2 * n; i *= 2) {
        if (tid % i == 0) {
            int k = i;
            if (n - tid < k) k = n - tid;
            for (int j = 0; j < k / 2; ++j) {
                Bufferfly(&nums[br(tid + j, bits)], &nums[br(tid + j + k / 2, bits)], Complex::W(k, j));
            }
        }
        __syncthreads();
    }
    result[tid] = nums[br(tid, bits)];
}

void printSequence(Complex nums[], const int N) {
    printf("[");
    for (int i = 0; i < N; ++i) {
        double real = nums[i].real, imag = nums[i].imag;
        if (imag == 0) printf("%.16f", real);
        else {
            if (imag > 0) printf("%.16f+%.16fi", real, imag);
            else printf("%.16f%.16fi", real, imag);
        }
        if (i != N - 1) printf(", ");
    }
    printf("]\n");
}

int main() {
    srand(time(0));
    const int TPB = 1024;
    const int N = 1024 * 32;
    const int bits = GetBits(N);

    Complex *nums = (Complex*)malloc(sizeof(Complex) * N), *dNums, *dResult;
    for (int i = 0; i < N; ++i) {
        nums[i] = Complex::GetRandomReal();
    }
    printf("Length of Sequence: %d\n", N);
    // printf("Before FFT: \n");
    // printSequence(nums, N);

    float s = GetTickCount();

    cudaMalloc((void**)&dNums, sizeof(Complex) * N);
    cudaMalloc((void**)&dResult, sizeof(Complex) * N);
    cudaMemcpy(dNums, nums, sizeof(Complex) * N, cudaMemcpyHostToDevice);
    
    dim3 threadPerBlock(TPB);
    dim3 blockNum((N + threadPerBlock.x - 1) / threadPerBlock.x);
    FFT<<<blockNum, threadPerBlock>>>(dNums, dResult, N, bits);

    cudaMemcpy(nums, dResult, sizeof(Complex) * N, cudaMemcpyDeviceToHost);

    float cost = GetTickCount() - s;
    // printf("After FFT: \n");
    // printSequence(nums, N);
    printf("Time of Transfromation: %fms", cost);

    free(nums);
    cudaFree(dNums);
    cudaFree(dResult);
}
