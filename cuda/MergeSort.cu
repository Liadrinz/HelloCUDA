// 归并排序
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <Windows.h>

__global__ void MergeSort(int *nums, int *temp, int n) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = 2; i < 2 * n; i *= 2) {
        int len = i;
        if (n - tid < len) len = n - tid;
        if (tid % i == 0) {
            int *seqA = &nums[tid], lenA = i / 2, j = 0;
            int *seqB = &nums[tid + lenA], lenB = len - lenA, k = 0;
            int p = tid;
            while (j < lenA && k < lenB) {
                if (seqA[j] < seqB[k]) {
                    temp[p++] = seqA[j++];
                } else {
                    temp[p++] = seqB[k++];
                }
            }
            while (j < lenA)
                temp[p++] = seqA[j++];
            while (k < lenB)
                temp[p++] = seqB[k++];
            for (int j = tid; j < tid + len; j++)
                nums[j] = temp[j];
        }
        __syncthreads();
    }
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

    // 拷贝到设备
    int *dNums;
    cudaMalloc((void**)&dNums, sizeof(int) * size);
    cudaMemcpy(dNums, nums, sizeof(int) * size, cudaMemcpyHostToDevice);

    // 临时存储
    int *dTemp;
    cudaMalloc((void**)&dTemp, sizeof(int) * size);

    dim3 threadPerBlock(1024);
    dim3 blockNum((size + threadPerBlock.x - 1) / threadPerBlock.x);
    MergeSort<<<blockNum, threadPerBlock>>>(dNums, dTemp, size);

    cudaMemcpy(nums, dNums, sizeof(int) * size, cudaMemcpyDeviceToHost);

    // 打印结果
    // for (int i = 0; i < size; ++i) {
    //     printf("%d ", nums[i]);
    // }
    // printf("\n");

    free(nums);
    cudaFree(dNums);
    cudaFree(dTemp);

    printf("Number of numbers: %d\n", size);
    printf("Sorting time: %fms\n", GetTickCount() - s);
}