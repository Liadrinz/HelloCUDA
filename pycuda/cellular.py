import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import matplotlib.pyplot as plt

from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void NextState(int mat[], int *m, int *n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int neighbor = 0;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            int nx = x + i, ny = y + j;
            if (nx >= 0 && nx < n[0] && ny >= 0 && ny < m[0]) {
                neighbor += mat[ny * n[0] + nx];
            }
        }
    }
    __syncthreads();
    if (neighbor == 3) {
        mat[y * n[0] + x] = 1;
    } else if (neighbor >= 4 || neighbor <= 1) {
        mat[y * n[0] + x] = 0;
    }
}
""")

next_state = mod.get_function("NextState")

def main():
    x, y = 1024, 1024
    mat = np.random.randint(2, size=[x * y])
    plt.ion()
    plt.show()
    plt.imshow(np.reshape(mat, [x, y]))
    plt.pause(0.1)
    for i in range(1000):
        next_state(
            drv.InOut(mat), drv.In(np.array([x])), drv.In(np.array([y])),
            block=(32, 32, 1), grid=((x + 32 - 1) // 32, (y + 32 - 1) // 32)
        )
        plt.cla()
        plt.imshow(np.reshape(mat, [x, y]))
        plt.pause(0.1)

if __name__ == '__main__':
    main()
