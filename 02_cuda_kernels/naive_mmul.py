import cupy as cp
import numpy as np
from cupyx.profiler import benchmark

naive_code = r'''
extern "C" __global__
void mmul(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}
'''

def main():
    mod = cp.RawModule(code=naive_code)
    mmul_kernel = mod.get_function('mmul')

    N = 2048
    a_gpu = cp.random.rand(N, N, dtype=cp.float32)
    b_gpu = cp.random.rand(N, N, dtype=cp.float32)
    c_gpu = cp.zeros((N, N), dtype=cp.float32)

    block_size = (16, 16)
    grid_size = (N // 16, N // 16)

    def run_mmul(a, b, c):
        mmul_kernel(grid_size, block_size, (a, b, c, N))
        return c

    _ = run_mmul(a_gpu, b_gpu, c_gpu)
    cp.cuda.Stream.null.synchronize()
    
    result = benchmark(run_mmul, (a_gpu, b_gpu, c_gpu), n_repeat=10, n_warmup=2)
    print("Naive Kernel 耗时:", result)

if __name__ == "__main__":
    main()
