"""Pre-flight check: confirm the GPU works and measure its matmul throughput.

Run:  python pretrain/check_env.py
"""

import time

import torch


def matmul_tflops(dtype: torch.dtype, n: int = 4096, iters: int = 20) -> float:
    """Time n x n matrix multiplications and return trillions of operations per second."""
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)

    # Warm-up: the first calls pay one-time setup costs we don't want to time.
    for _ in range(3):
        a @ b

    # GPU work is queued asynchronously, so wait for it to finish before reading the clock.
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start

    # Each of the n*n outputs needs n multiplies and n adds: 2 * n^3 operations per matmul.
    flops = 2 * n**3 * iters
    return flops / seconds / 1e12


def main() -> None:
    print(f"torch {torch.__version__}")
    if not torch.cuda.is_available():
        print("CUDA not available: training would run on the CPU.")
        return

    props = torch.cuda.get_device_properties(0)
    print(f"gpu: {props.name}, {props.total_memory / 1024**3:.1f} GB")
    print(f"compute capability: {props.major}.{props.minor}")

    for dtype in (torch.float32, torch.float16):
        print(f"{dtype!s:>14} matmul: {matmul_tflops(dtype):6.2f} TFLOPS")


if __name__ == "__main__":
    main()
