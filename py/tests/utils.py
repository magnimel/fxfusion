import time
import torch

def check_correctness(engine, model, x, rtol=1e-3, atol=1e-3):
    cpp_output = engine.run([x])[0]
    py_output = model(x)

    if cpp_output.shape != py_output.shape:
        return False, f"shape mismatch — cpp: {cpp_output.shape}, py: {py_output.shape}"

    if torch.allclose(cpp_output, py_output, rtol=rtol, atol=atol):
        return True, ""

    diff = (cpp_output - py_output).abs()
    return False, f"max diff: {diff.max().item():.6f}, mean diff: {diff.mean().item():.6f}"


def benchmark(name, fn, warmup=50, iters=1000):
    for _ in range(warmup):
        fn()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    end = time.perf_counter()

    ms = (end - start) * 1000 / iters
    print(f"{name:<13}: {ms:.4f} ms")
    return ms