# GPU Matrix Multiplication & Kernel Optimization Benchmark

本项目记录了基于 CuPy 和 NVRTC 进行 GPU 科学计算性能优化的完整实践，重点对比了高层 API 调用与手写底层 CUDA Kernel 的性能差异。

## 🖥️ 实验环境
- **硬件**: NVIDIA GeForce RTX 4060 Laptop GPU
- **软件**: Windows, CUDA Toolkit 12.6, CuPy 14.2, Python
- **测试规模**: 矩阵乘法 (2048x2048, float32)

## 🚀 核心实验数据与结论

### 1. 异步执行与并发边界测试 (Stream)
- 在 1500x1500 float32 矩阵乘法中，利用双流（Stream）并发较串行实现约 **45 倍**加速。
- 在 4000x4000 float64 场景下，验证因 GPU 算力饱和导致并发无收益，明确了 Stream 并发的适用边界。

### 2. 图像滤波与 Kernel 融合 (Fuse)
- **显存开销**：基于 `cupyx.scipy.ndimage.gaussian_filter` 验证了 float32 较 float64 有约 **2.4 倍**的性能提升。
- **Kernel 融合**：利用 `@cp.fuse()` 消除中间数组显存读写，实测 GPU 耗时从 1.8 ms 降至 0.27 ms（**6.8 倍**加速）。

### 3. 手写 C++ CUDA Kernel 优化 (RawKernel)
通过 CuPy 的 `RawModule` 直接注入 C++ CUDA 代码，绕过繁琐的本地 MSVC 编译环境配置。
- **Naive Baseline (未优化)**: 22.8 ms。每个线程计算一个元素，存在大量全局内存读取。
- **Shared Memory Tiled (分块优化)**: 16.3 ms。引入 `__shared__` 和 `__syncthreads()`，减少了全局访存。
- **官方 cuBLAS 库 (`cp.dot`)**: ~2 ms。

**技术反思**：
手写 Kernel 虽然实现了逻辑正确（误差 0.00000），但由于未使用寄存器分块和向量化读取，在 2048 规模下依然受限于显存带宽（Memory Bound）。这让我深刻理解了 NVIDIA 官方库底层优化的含金量。

## 🛠️ 如何运行
确保已安装带有 CUDA 支持的 CuPy (`pip install cupy-cuda12x`)，在对应目录下直接运行 Python 脚本即可。
## 🚀 高阶挑战：大模型核心算子 FlashAttention (Triton 实现)

利用 Triton 实现了带有 Online Softmax 机制的融合 Scaled Dot-Product Attention 算子。
通过将矩阵乘法、缩放、Softmax 融合在单个 Kernel 中，并在 SRAM 中增量更新，彻底消除了 \(N \times N\) 中间矩阵的显存开销。

### ⚡ 性能对比 (Batch=4, Head=8, SeqLen=1024, Dim=64, FP16)
| 算子实现版本 | 耗时 | 优化说明 |
| :--- | :--- | :--- |
| PyTorch 官方 (`scaled_dot_product_attention`) | 0.573 ms | 原生实现 |
| **手写 Triton 融合 Attention** | **0.477 ms** | **Online Softmax，消除中间显存读写，性能反超官方库！** |
