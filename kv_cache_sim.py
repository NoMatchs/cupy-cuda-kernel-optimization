import torch
import time


def simulate_traditional_kv_cache(max_length=2048, num_heads=32, head_dim=128):
    # 假设数据类型是 float16 (2 bytes)
    # 传统方式：一次性预先分配最大长度的显存
    # 形状: [batch_size, num_heads, max_length, head_dim]
    shape = (1, num_heads, max_length, head_dim)
    element_size = 2
    expected_size = (
        shape[0] * shape[1] * shape[2] * shape[3] * element_size / (1024**2)
    )  # MB

    print(f"传统 KV Cache 预分配最大长度 {max_length} 需要: {expected_size:.2f} MB")

    # 模拟 10 个用户同时使用，但每个用户实际上只说了几十个词就结束了
    # 如果按照传统方式，显卡必须为这 10 个用户分配满 2048 长度的显存
    batch_size = 10
    total_allocated = expected_size * batch_size
    print(f"10 个并发请求，传统方式强制分配显存: {total_allocated:.2f} MB")

    # 实际使用量可能只有 1/10
    print(f"如果实际平均只用 200 长度，浪费显存高达: {total_allocated * 0.9:.2f} MB")


# 运行一下看看
simulate_traditional_kv_cache()
