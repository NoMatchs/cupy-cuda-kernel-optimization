import cupy as cp
import numpy as np
from cupyx.profiler import benchmark

tiled_kernel_code = r'''
#define BLOCK_SIZE 16

extern "C" __global__
void mmul_tiled_opt(const float* A, const float* B, float* C, int N) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + 1]; 

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float sum = 0.0f;

    for (int m = 0; m < N / BLOCK_SIZE; ++m) {
        As[ty][tx] = A[row * N + m * BLOCK_SIZE + tx];
        Bs[ty][tx] = B[(m * BLOCK_SIZE + ty) * N + col];
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}
'''

def main():
    mod = cp.RawModule(code=tiled_kernel_code)
    mmul_tiled_opt = mod.get_function('mmul_tiled_opt')

    N = 2048
    a_gpu = cp.random.rand(N, N, dtype=cp.float32)
    b_gpu = cp.random.rand(N, N, dtype=cp.float32)
    c_gpu = cp.zeros((N, N), dtype=cp.float32)

    block_size = (16, 16)
    grid_size = (N // 16, N // 16)

    def run_opt(a, b, c):
        mmul_tiled_opt(grid_size, block_size, (a, b, c, N))
        return c

    _ = run_opt(a_gpu, b_gpu, c_gpu)
    cp.cuda.Stream.null.synchronize()
    
    expected = cp.dot(a_gpu, b_gpu)
    print(f"误差: {cp.max(cp.abs(c_gpu - expected)):.5f}")
    
    result = benchmark(run_opt, (a_gpu, b_gpu, c_gpu), n_repeat=10, n_warmup=2)
    print("Tiled Kernel 耗时:", result)

if __name__ == "__main__":
    main()
