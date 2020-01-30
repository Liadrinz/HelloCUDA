// 秩排序
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <Windows.h>

__global__ void Rank(int nums[], int sorted[], int n) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n) return;  // 防止冗余线程计算出相同的秩覆盖结果
    int rank = 0;
    for (int i = 0; i < n; ++i) {
        if (nums[tid] > nums[i] || nums[tid] == nums[i] && tid > i) rank++;
    }
    sorted[rank] = nums[tid];
}

int main() {

    float s = GetTickCount();

    // 初始化数列
    int size = 1024 * 1024;
    int *nums = (int*)malloc(sizeof(int) * size);
    srand(time(0));
    for (int i = 0; i < size; ++i) {
        nums[i] = rand();
    }

    // 打印输入
    // for (int i = 0; i < size; ++i) {
    //     printf("%d ", nums[i]);
    // }
    // printf("\n");

    // 拷贝到设备
    int *dNums;
    cudaMalloc((void**)&dNums, sizeof(int) * size);
    cudaMemcpy(dNums, nums, sizeof(int) * size, cudaMemcpyHostToDevice);

    int *dSorted;
    cudaMalloc((void**)&dSorted, sizeof(int) * size);

    dim3 threadPerBlock(1024);
    dim3 blockNum((size + threadPerBlock.x - 1) / threadPerBlock.x);
    Rank<<<blockNum, threadPerBlock>>>(dNums, dSorted, size);

    // 结果拷贝回主机
    cudaMemcpy(nums, dSorted, sizeof(int) * size, cudaMemcpyDeviceToHost);

    // 打印结果
    // for (int i = 0; i < size; ++i) {
    //     printf("%d ", nums[i]);
    // }
    // printf("\n");

    printf("Number of numbers: %d\n", size);
    printf("Sorting time: %fms\n", GetTickCount() - s);
    free(nums);
    cudaFree(dNums);
    cudaFree(dSorted);
}