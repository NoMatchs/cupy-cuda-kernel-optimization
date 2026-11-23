import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cndi
import scipy.ndimage as ndi
from cupyx.profiler import benchmark

def main():
    img_cpu = np.random.rand(2048, 2048).astype(np.float32)
    img_gpu = cp.asarray(img_cpu)
    sigma = 3.0

    def cpu_filter(img): return ndi.gaussian_filter(img, sigma=sigma)
    def gpu_filter(img): return cndi.gaussian_filter(img, sigma=sigma)

    print("CPU (SciPy): ", benchmark(cpu_filter, (img_cpu,), n_repeat=20, n_warmup=5))
    print("GPU (仅计算):", benchmark(gpu_filter, (img_gpu,), n_repeat=20, n_warmup=5))

if __name__ == "__main__":
    main()
