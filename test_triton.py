import torch
import triton
import triton.language as tl
import time


# ==========================================
# 1. 环境验证 (加法)
# ==========================================
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


n = 1024
x = torch.rand(n, device="cuda")
y = torch.rand(n, device="cuda")
out = torch.empty_like(x)
grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
add_kernel[grid](x, y, out, n, BLOCK=256)
print("Triton 环境正常" if torch.allclose(out, x + y) else "结果错误")


# ==========================================
# 2. 矩阵乘法 Kernel
# ==========================================
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid // num_pid_m
    pid_n = pid % num_pid_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float32)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
    )
    return c


# ==========================================
# 3. 科学测速 (使用 CUDA Event 计时)
# ==========================================
if __name__ == "__main__":
    M = N = K = 2048
    a = torch.rand((M, K), device="cuda", dtype=torch.float32)
    b = torch.rand((K, N), device="cuda", dtype=torch.float32)

    print(f"正在测试 Triton 矩阵乘法 ({M}x{K} * {K}x{N})...")

    # 预热 (非常重要，避免首次编译或显存分配影响)
    for _ in range(3):
        _ = triton_matmul(a, b)
    torch.cuda.synchronize()

    # 验证结果正确性 (用 PyTorch 官方矩阵乘法作对比)
    c_triton = triton_matmul(a, b)
    c_torch = torch.matmul(a, b)
    error = torch.max(torch.abs(c_triton - c_torch)).item()
    print(f"结果误差: {error:.5f}")

    # 用 CUDA Event 计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    iterations = 10
    start_event.record()
    for _ in range(iterations):
        _ = triton_matmul(a, b)
    end_event.record()

    torch.cuda.synchronize()
    avg_time_ms = start_event.elapsed_time(end_event) / iterations
    print(f"Triton 平均 GPU 耗时: {avg_time_ms:.3f} ms")
