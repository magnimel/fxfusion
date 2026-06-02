import time
import torch
import torch.nn as nn

def check_correctness(engine, model, x):
    cpp_output = engine.run([x])[0]
    py_output = model(x)

    if cpp_output.shape != py_output.shape:
        print("[False] shape mismatch")
        print("cpp:", cpp_output.shape)
        print("py :", py_output.shape)
        return False

    if torch.allclose(cpp_output, py_output, rtol=1e-4, atol=1e-5):
        print("[True] Success")
        return True

    diff = (cpp_output - py_output).abs()
    print("[False] elements mismatch")
    print("max diff :", diff.max().item())
    print("mean diff:", diff.mean().item())
    return False


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


