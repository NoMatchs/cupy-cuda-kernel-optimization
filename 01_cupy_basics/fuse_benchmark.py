import cupy as cp
from cupyx.profiler import benchmark

def unfused(x, y):
    return cp.sqrt(cp.exp(x) + cp.sin(y)) * cp.tanh(x - y)

@cp.fuse()
def fused(x, y):
    return cp.sqrt(cp.exp(x) + cp.sin(y)) * cp.tanh(x - y)

def main():
    x = cp.random.rand(2048, 2048, dtype=cp.float32)
    y = cp.random.rand(2048, 2048, dtype=cp.float32)
    
    print("Unfused:", benchmark(unfused, (x, y), n_repeat=50, n_warmup=10))
    print("Fused:  ", benchmark(fused, (x, y), n_repeat=50, n_warmup=10))

if __name__ == "__main__":
    main()
