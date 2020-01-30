// PI的计算
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <Windows.h>

const long double PI_48 = 3.141592653589793238462643;
const int N = 1024 * 1024;  // 积分区间个数
const int TPB = 1024;  // 每块线程数

double sum(int n, double *nums) {
    if (n == 0) return 0;
    if (n == 1) return nums[0];
    return sum(n / 2, nums) + sum(n - n / 2, &nums[n / 2]);
}

__device__ double f(double x) {
    return 4.0 / (1.0 + x * x);
}

__global__ void IntervalArea(int n, double *integral) {
    __shared__ double cache[TPB];
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int cid = threadIdx.x;
    int bid = blockIdx.x;
    double x;
    x = (tid + 0.5) / n;
    cache[cid] = f(x) / n;
    __syncthreads();

    // 归约
    for (int i = 2; i < 2 * TPB; i *= 2) {
        if (cid % i == 0) {
            cache[cid] += cache[cid + i / 2];
        }
        __syncthreads();
    }

    if (cid == 0)
        integral[bid] = cache[0];
}

int main() {
    float s = GetTickCount();

    dim3 threadPerBlock(TPB);
    dim3 blockNum((N + threadPerBlock.x - 1) / threadPerBlock.x);
    double *area = (double*)malloc(sizeof(double) * blockNum.x);
    double *dArea;
    cudaMalloc((void**)&dArea, sizeof(double) * blockNum.x);
    IntervalArea<<<blockNum, threadPerBlock>>>(N, dArea);
    cudaMemcpy(area, dArea, sizeof(double) * blockNum.x, cudaMemcpyDeviceToHost);
    long double pi = (long double)sum(blockNum.x, area);
    long double error = PI_48 - pi;
    printf("Integral Intervals: %d\n", N);
    printf("Result of PI: %.48f\n", pi);
    printf("Computing Time: %fms\n", GetTickCount() - s);
    printf("Computing Error: %.48e\n", error);
    free(area);
    cudaFree(dArea);
}