import torch
import triton
import triton.language as tl


@triton.jit
def paged_kv_gather_kernel(
    cache_ptr,
    block_table_ptr,
    out_ptr,
    seq_len,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # 每个 program 负责读取一个物理块
    pid = tl.program_id(0)

    # 通过 block_table (页表) 找到真正的物理块号
    physical_block_id = tl.load(block_table_ptr + pid)

    # 计算物理地址偏移量
    offs = tl.arange(0, BLOCK_SIZE * HEAD_DIM)
    # 从分散的显存块中抓取数据
    data = tl.load(cache_ptr + physical_block_id * BLOCK_SIZE * HEAD_DIM + offs)

    # 写入到连续的输出缓冲区
    tl.store(out_ptr + pid * BLOCK_SIZE * HEAD_DIM + offs, data)


# 测试一下
seq_len = 1024
head_dim = 64
block_size = 16
num_blocks = seq_len // block_size

# 模拟：准备 100 个物理块的显存池
cache_pool = torch.randn(100, block_size, head_dim, device="cuda", dtype=torch.float16)
# 乱序分配（模拟显存碎片）
block_table = torch.randint(0, 100, (num_blocks,), device="cuda", dtype=torch.int32)
out = torch.empty(num_blocks, block_size, head_dim, device="cuda", dtype=torch.float16)

# 启动 Kernel
paged_kv_gather_kernel[(num_blocks,)](
    cache_pool,
    block_table,
    out,
    seq_len,
    head_dim,
    BLOCK_SIZE=block_size,
    HEAD_DIM=head_dim,
)
print("分页 KV Cache 数据重组成功！")
