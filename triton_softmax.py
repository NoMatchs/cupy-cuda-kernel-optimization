import torch
import triton
import triton.language as tl
import torch.utils.benchmark as benchmark


# ==========================================
# Triton Softmax Kernel
# ==========================================
@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    # 获取当前处理的行索引
    row_idx = tl.program_id(0)

    # 指向当前行的开头
    row_start_ptr = input_ptr + row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 加载一整行数据，用 -inf 填充超出长度的部分
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))

    # 1. 减去最大值防溢出 (Max)
    row_minus_max = row - tl.max(row, axis=0)

    # 2. 计算指数 (Exp)
    numerator = tl.exp(row_minus_max)

    # 3. 计算分母 (Sum)
    denominator = tl.sum(numerator, axis=0)

    # 4. 归一化并写回
    softmax_output = numerator / denominator
    output_ptrs = output_ptr + row_idx * n_cols + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def triton_softmax(x):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    # BLOCK_SIZE 必须是 2 的幂，且大于等于 n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # 每一行启动一个 program
    softmax_kernel[(n_rows,)](x, output, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return output


# ==========================================
# 验证与测速
# ==========================================
if __name__ == "__main__":
    # 模拟大模型里的 Attention 矩阵，比如 1024 行，每行 4096 个特征
    n_rows, n_cols = 1024, 4096
    x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)

    print(f"正在测试 Softmax ({n_rows}x{n_cols})...")

    # 验证正确性
    out_triton = triton_softmax(x)
    out_torch = torch.softmax(x, dim=-1)
    error = torch.max(torch.abs(out_triton - out_torch)).item()
    print(f"结果误差: {error:.6f} (通常应 < 0.00001)")

    # 测速
    def run_triton():
        triton_softmax(x)

    def run_torch():
        torch.softmax(x, dim=-1)

    # 预热
    run_triton()
    run_torch()
    torch.cuda.synchronize()

    # 官方计时器
    t_triton = benchmark.Timer(stmt="run_triton()", globals=globals()).timeit(100)
    t_torch = benchmark.Timer(stmt="run_torch()", globals=globals()).timeit(100)

    print(f"PyTorch 官方 Softmax 耗时: {t_torch.mean * 1000:.3f} ms")
    print(f"手写 Triton Softmax 耗时:  {t_triton.mean * 1000:.3f} ms")
