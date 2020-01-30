// 矩阵乘法
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <Windows.h>

__global__ void MatMul(float *A, float *B, float *C, int width) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int sum = 0;
    for (int k = 0; k < width; k++) {
        int a = A[j * width + k];
        int b = B[k * width + i];
        sum += a * b;
    }
    C[j * width + i] = sum;
}

int main() {

    float s = GetTickCount();

    // 假设A是(m, n), B是(n, m)
    int m = 1024, n = 1024;

    // CPU分配内存
    float *A, *B, *C;
    int memsize = sizeof(float) * m * n;
    int ressize =  sizeof(float) * m * m;
    A = (float*)malloc(memsize);
    B = (float*)malloc(memsize);
    C = (float*)malloc(ressize);

    // 初始化
    for (int i = 0; i < m * n; ++i) {
        A[i] = 1;
        B[i] = 1;
    }

    // GPU分配内存
    float *dA, *dB, *dC;
    cudaMalloc((void**)&dA, memsize);
    cudaMalloc((void**)&dB, memsize);
    cudaMalloc((void**)&dC, ressize);

    // 拷贝数据到设备
    cudaMemcpy(dA, A, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, ressize, cudaMemcpyHostToDevice);

    // 调用kernel
    dim3 threadPerBlock(16, 16);
    dim3 blockNum(n / threadPerBlock.x, m / threadPerBlock.y);
    MatMul<<<blockNum, threadPerBlock>>>(dA, dB, dC, n);

    // 拷贝结果到主机
    cudaMemcpy(C, dC, ressize, cudaMemcpyDeviceToHost);

    // 打印结果
    // for (int i = 0; i < m; ++i) {
    //     for (int j = 0; j < m; ++j) {
    //         printf("%.1f ", C[j * m + i]);
    //     }
    //     printf("\n");
    // }

    // 释放内存
    free(A);
    free(B);
    free(C);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    printf("Computing Time: %fms\n", GetTickCount() - s);
}