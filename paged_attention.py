import torch
import triton
import triton.language as tl
import torch.utils.benchmark as benchmark


@triton.jit
def paged_attention_kernel(
    Q,
    K_cache,
    V_cache,
    Out,
    block_table,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vb,
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
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # 获取 batch 和 head 的索引
    batch_idx = off_hz // H
    head_idx = off_hz % H

    # 计算 Q 和 Out 的内存偏移
    q_offset = batch_idx * stride_qz + head_idx * stride_qh
    o_offset = batch_idx * stride_oz + head_idx * stride_oh

    # 加载 Q 块
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(q_ptrs)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        logical_block_idx = start_n // BLOCK_SIZE
        physical_block_idx = tl.load(block_table + logical_block_idx)

        offs_n = tl.arange(0, BLOCK_N)

        # 修复：K_cache 和 V_cache 的物理偏移量计算
        # 物理地址 = 物理块号*KB维度步长 + head_idx*H维度步长 + 块内偏移
        k_ptrs = (
            K_cache
            + physical_block_idx * stride_kb
            + head_idx * stride_kh
            + offs_n[None, :] * stride_kn
            + offs_k[:, None] * stride_kk
        )
        v_ptrs = (
            V_cache
            + physical_block_idx * stride_vb
            + head_idx * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_k[None, :] * stride_vk
        )

        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        # 修复：强制使用 IEEE 精度，消除 TF32 误差
        qk = tl.dot(q, k, input_precision="ieee") * (1.0 / (BLOCK_DMODEL**0.5))

        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        qk = qk - m_new[:, None]
        p = tl.exp(qk)
        l_i = l_i * alpha + tl.sum(p, 1)

        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.float16), v, acc, input_precision="ieee")
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
    tl.store(o_ptrs, acc.to(tl.float16))


def run_paged_attention(q, k_cache, v_cache, block_table):
    Z, H, N_CTX, D = q.shape
    out = torch.empty_like(q)
    grid = (triton.cdiv(N_CTX, 64), Z * H)
    BLOCK_SIZE = 16

    paged_attention_kernel[grid](
        q,
        k_cache,
        v_cache,
        out,
        block_table,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        Z,
        H,
        N_CTX,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_M=64,
        BLOCK_N=16,
        BLOCK_DMODEL=D,
    )
    return out


if __name__ == "__main__":
    Z, H, N_CTX, D = 4, 8, 1024, 64
    BLOCK_SIZE = 16

    q = torch.randn((Z, H, N_CTX, D), device="cuda", dtype=torch.float16)

    k_pool = torch.randn(100, H, BLOCK_SIZE, D, device="cuda", dtype=torch.float16)
    v_pool = torch.randn(100, H, BLOCK_SIZE, D, device="cuda", dtype=torch.float16)

    num_blocks_needed = N_CTX // BLOCK_SIZE
    block_table = torch.randperm(100, device="cuda", dtype=torch.int32)[
        :num_blocks_needed
    ]

    k_real = torch.empty_like(q)
    v_real = torch.empty_like(q)
    for i in range(num_blocks_needed):
        phys_id = block_table[i].item()
        # 修复：正确的广播逻辑
        k_real[:, :, i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE, :] = (
            k_pool[phys_id].unsqueeze(0).expand(Z, -1, -1, -1)
        )
        v_real[:, :, i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE, :] = (
            v_pool[phys_id].unsqueeze(0).expand(Z, -1, -1, -1)
        )

    print(f"正在测试 Paged Attention (Batch={Z}, Head={H}, SeqLen={N_CTX}, Dim={D})...")

    out_paged = run_paged_attention(q, k_pool, v_pool, block_table)
    out_torch = torch.nn.functional.scaled_dot_product_attention(q, k_real, v_real)

    error = torch.max(torch.abs(out_paged - out_torch)).item()
    print(f"结果误差: {error:.5f}")

    def run_paged():
        run_paged_attention(q, k_pool, v_pool, block_table)

    def run_torch():
        torch.nn.functional.scaled_dot_product_attention(q, k_real, v_real)

    run_paged()
    run_torch()
    torch.cuda.synchronize()

    t_paged = benchmark.Timer(stmt="run_paged()", globals=globals()).timeit(50)
    t_torch = benchmark.Timer(stmt="run_torch()", globals=globals()).timeit(50)

    print(f"PyTorch 连续显存 Attention 耗时: {t_torch.mean * 1000:.3f} ms")
    print(f"手写 Triton 分页 Attention 耗时:  {t_paged.mean * 1000:.3f} ms")
