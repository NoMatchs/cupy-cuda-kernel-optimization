import cupy as cp
import time
from cupyx.profiler import benchmark

def main():
    a = cp.random.rand(1500, 1500, dtype=cp.float32)
    b = cp.random.rand(1500, 1500, dtype=cp.float32)
    c = cp.random.rand(1500, 1500, dtype=cp.float32)
    d = cp.random.rand(1500, 1500, dtype=cp.float32)

    s1, s2 = cp.cuda.Stream(), cp.cuda.Stream()
    start = cp.cuda.Event(); end = cp.cuda.Event()

    start.record()
    with s1: r1 = cp.dot(a, b)
    with s2: r2 = cp.dot(c, d)
    end.record()
    end.synchronize()

    print(f"双流并发 GPU 耗时: {cp.cuda.get_elapsed_time(start, end):.3f} ms")

if __name__ == "__main__":
    main()
