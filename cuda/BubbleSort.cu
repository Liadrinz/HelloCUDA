// 奇偶交换排序(并行冒泡排序)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <Windows.h>

__global__ void Switch(int nums[], int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    for (int p = 0; p < n; ++p) {
        if ((i - p) % 2 && i < n - 1 && nums[i] > nums[i + 1]) {
            int temp = nums[i];
            nums[i] = nums[i + 1];
            nums[i + 1] = temp;
        }
        __syncthreads();
    }
}

// 串行冒泡排序
void SequentialBubbleSort(int nums[], int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size - 1; ++j) {
            if (nums[j] > nums[j + 1]) {
                int temp = nums[j];
                nums[j] = nums[j + 1];
                nums[j + 1] = temp;
            }
        }
    }
}

// 奇偶交换排序
void OddEvenSwitchSort(int nums[], int size) {
    // 拷贝到设备
    int *dNums;
    cudaMalloc((void**)&dNums, sizeof(int) * size);
    cudaMemcpy(dNums, nums, sizeof(int) * size, cudaMemcpyHostToDevice);

    // 进行size次奇偶交换
    dim3 threadPerBlock(1024);
    dim3 blockNum((size + threadPerBlock.x - 1) / threadPerBlock.x);
    Switch<<<blockNum, threadPerBlock>>>(dNums, size);

    // 结果拷贝回主机
    cudaMemcpy(nums, dNums, sizeof(int) * size, cudaMemcpyDeviceToHost);

    // 打印结果
    // for (int i = 0; i < size; ++i) {
    //     printf("%d ", nums[i]);
    // }
    cudaFree(dNums);
}

int main() {

    float s = GetTickCount();

    // 初始化数列
    int size = 65536;
    int *nums1 = (int*)malloc(sizeof(int) * size), *nums2 = (int*)malloc(sizeof(int) * size);
    srand(time(0));
    for (int i = 0; i < size; ++i) {
        int num = rand();
        nums1[i] = num;
        nums2[i] = num;
    }
    printf("Number of numbers: %d\n", size);
    OddEvenSwitchSort(nums1, size);
    float tp = GetTickCount() - s;
    printf("Parallel Sorting Time: %fms\n", tp);
    s = GetTickCount();
    SequentialBubbleSort(nums2, size);
    float ts = GetTickCount() - s;
    printf("Sequential Sorting Time: %fms\n", ts);
    printf("Acceleration Factor: %f\n", ts / tp);
    free(nums1);
    free(nums2);
}