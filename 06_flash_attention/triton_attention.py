import torch
import triton
import triton.language as tl
import torch.utils.benchmark as benchmark


# ==========================================
# Triton 融合 Attention Kernel
# ==========================================
@triton.jit
def attention_kernel(
    Q,
    K,
    V,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh

    # 加载 Q 块 (BLOCK_M x BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(q_ptrs)

    # 初始化累加器和 Softmax 的统计量
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # 循环遍历 K 和 V 的块
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # 加载 K 块并转置，计算 Q * K^T
        k_ptrs = (
            K + qvk_offset + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
        )
        k = tl.load(k_ptrs)
        qk = tl.dot(q, k)  # 这里的 tl.dot 会自动利用 Tensor Core

        # 缩放点积
        qk = qk * (1.0 / (BLOCK_DMODEL**0.5))

        # 核心：在线计算 Softmax (Online Softmax)
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        qk = qk - m_new[:, None]
        p = tl.exp(qk)
        l_i = l_i * alpha + tl.sum(p, 1)

        # 更新累加器
        acc = acc * alpha[:, None]

        # 加载 V 块并计算 P * V
        v_ptrs = (
            V + qvk_offset + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        )
        v = tl.load(v_ptrs)
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_new

    # 归一化并写回
    acc = acc / l_i[:, None]
    o_ptrs = (
        Out + qvk_offset + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
    )
    tl.store(o_ptrs, acc.to(tl.float16))


# ==========================================
# 运行与对比测试
# ==========================================
def run_triton_attention(q, k, v):
    Z, H, N_CTX, D = q.shape
    out = torch.empty_like(q)
    grid = (triton.cdiv(N_CTX, 64), Z * H)
    attention_kernel[grid](
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        Z,
        H,
        N_CTX,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_DMODEL=D,
    )
    return out


if __name__ == "__main__":
    # 模拟大模型输入：Batch=4, Head=8, SeqLen=1024, HeadDim=64 (半精度)
    Z, H, N_CTX, D = 4, 8, 1024, 64
    q = torch.randn((Z, H, N_CTX, D), device="cuda", dtype=torch.float16)
    k = torch.randn((Z, H, N_CTX, D), device="cuda", dtype=torch.float16)
    v = torch.randn((Z, H, N_CTX, D), device="cuda", dtype=torch.float16)

    print(f"正在测试 Attention (Batch={Z}, Head={H}, SeqLen={N_CTX}, Dim={D})...")

    # 1. 验证正确性
    out_triton = run_triton_attention(q, k, v)
    out_torch = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    error = torch.max(torch.abs(out_triton - out_torch)).item()
    print(f"结果误差: {error:.5f}")

    # 2. 性能计时
    def run_triton():
        run_triton_attention(q, k, v)

    def run_torch():
        torch.nn.functional.scaled_dot_product_attention(q, k, v)

    # 预热
    run_triton()
    run_torch()
    torch.cuda.synchronize()

    t_triton = benchmark.Timer(stmt="run_triton()", globals=globals()).timeit(50)
    t_torch = benchmark.Timer(stmt="run_torch()", globals=globals()).timeit(50)

    print(f"PyTorch 官方 Attention 耗时: {t_torch.mean * 1000:.3f} ms")
    print(f"手写 Triton Attention 耗时:  {t_triton.mean * 1000:.3f} ms")
