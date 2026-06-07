import csv
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS       = PROJECT_ROOT / "assets"
IMGS         = ASSETS / "imgs"
RESULTS_DIR  = ASSETS / "results"

ASSETS.mkdir(exist_ok=True)
IMGS.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------------

COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def benchmark(name: str, fn, device: str = "cpu", warmup: int = 100, iters: int = 5000) -> float:
    for _ in range(warmup):
        fn()

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available") 
        
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event   = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iters):
            fn()
        end_event.record()
        torch.cuda.synchronize()
        ms = start_event.elapsed_time(end_event) / iters
    else:
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        end = time.perf_counter()
        ms = (end - start) * 1000 / iters

    print(f"{name:<15}: {ms:.4f} ms")
    return ms

# -----------------------------------------------------------------------------
# Chart
# -----------------------------------------------------------------------------

def save_chart(label: str, results: dict[str, float], filename: str):
    names = list(results.keys())
    times = list(results.values())

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, times, color=COLORS[:len(names)])
    ax.set_ylabel("Latency (ms)")
    ax.set_title(label, pad=10, fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(times) * 1.20)
    ax.bar_label(bars, fmt="%.3f ms", padding=3)
    plt.tight_layout()

    path = IMGS / f"{filename}.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved {path}")

# -----------------------------------------------------------------------------
# CSV
# -----------------------------------------------------------------------------

def save_csv(filename: str, label: str, device: str, results: dict[str, float]):
    path = RESULTS_DIR / f"{filename}.csv"
    write_header = not path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["workload", "device", "backend", "latency_ms", "fxfusion_speedup"])

        fxfusion_ms = results.get("FXFusion", 1.0)
        for backend, ms in results.items():
            writer.writerow([
                label,
                device,
                backend,
                f"{ms:.4f}",
                f"{ms / fxfusion_ms:.2f}" if ms > 0 else "inf",
            ])

    print(f"Saved {path}")

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

def make_mlp(depth: int, width: int, device: str = "cpu") -> nn.Module:
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(width, width), nn.ReLU()]
    return nn.Sequential(*layers).eval().to(device)