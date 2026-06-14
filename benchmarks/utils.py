import csv
import sys
import textwrap
import time
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

BENCHMARK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_ROOT.parent
PY_ROOT = REPO_ROOT / "py"

sys.path.insert(0, str(PY_ROOT))
sys.path.insert(0, str(REPO_ROOT))

ASSETS = BENCHMARK_ROOT / "assets"
IMGS = ASSETS / "imgs"
RESULTS_DIR = ASSETS / "results"

ASSETS.mkdir(exist_ok=True)
IMGS.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


from fxfusion.engine import Engine
from fxfusion.models.transformer.models.gpt import GPT
from fxfusion.models.transformer.inference import (
    greedy_decode_static,
    engine_decode_static,
)
from fxfusion.models.transformer.layers.masks import (
    StaticDecoderMaskBuilder,
    make_static_buffer,
)
from tests.utils import check_correctness, compare_outputs


# -----------------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------------

COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def benchmark(
    name: str,
    fn,
    device: str = "cpu",
    warmup: int = 100,
    iters: int = 5000,
) -> float:
    for _ in range(warmup):
        fn()

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

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

    wrapped_label = "\n".join(textwrap.wrap(label, width=48))

    fig, ax = plt.subplots(figsize=(6, 4.5))

    bars = ax.bar(names, times, color=COLORS[:len(names)])

    ax.set_ylabel("Latency (ms)")
    ax.set_title(wrapped_label, pad=10, fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(times) * 1.25)

    ax.bar_label(bars, fmt="%.3f ms", padding=3)

    plt.tight_layout()

    path = IMGS / f"{filename}.png"

    plt.savefig(path, dpi=200)
    plt.close()

    print(f"Saved {path}")

# -----------------------------------------------------------------------------
# CSV
# -----------------------------------------------------------------------------

def save_csv(
    filename: str,
    label: str,
    device: str,
    results: dict[str, float],
):
    path = RESULTS_DIR / f"{filename}.csv"
    write_header = not path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(["workload","device","backend","latency_ms","fxfusion_speedup"])
            
        fxfusion_ms = results.get("FXFusion", 1.0)

        for backend, ms in results.items():
            writer.writerow([label,device,backend,f"{ms:.4f}",f"{ms / fxfusion_ms:.2f}" if ms > 0 else "inf"])

    print(f"Saved {path}")

def record_results(
    label: str,
    filename: str,
    device: str,
    results: dict[str, float],
):
    save_chart(label, results, f"cpu_{filename}")
    save_csv(f"cpu_{filename}", label, device, results)

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

def make_mlp(
    depth: int,
    width: int,
    device: str = "cpu",
) -> nn.Module:
    layers = []

    for _ in range(depth):
        layers += [
            nn.Linear(width, width),
            nn.ReLU(),
        ]

    return nn.Sequential(*layers).eval().to(device)


def make_gpt(
    d_model: int,
    h: int,
    vocab_size: int,
    expansion_factor: int,
    dropout: float,
    nx: int,
    device: str = "cpu",
) -> nn.Module:
    return GPT(
        d_model=d_model,
        h=h,
        vocab_size=vocab_size,
        expansion_factor=expansion_factor,
        dropout=dropout,
        Nx=nx,
    ).eval().to(device)


def make_resnet(
    version: int,
    device: str = "cpu",
) -> nn.Module:
    version = int(version)

    if version == 18:
        from torchvision.models import resnet18
        return resnet18(weights=None).eval().to(device)

    if version == 50:
        from torchvision.models import resnet50
        return resnet50(weights=None).eval().to(device)

    raise ValueError(f"Unsupported ResNet version: {version}. Supported: 18, 50")


# -----------------------------------------------------------------------------
# Shared Forward Benchmark Runner
# -----------------------------------------------------------------------------

def run_model_forward(
    label: str,
    model_name: str,
    model: nn.Module,
    inputs: list[torch.Tensor],
    filename: str,
    device: str = "cpu",
    warmup: int = 100,
    iters: int = 5000,
    correctness_atol: float = 1e-3,
    correctness_rtol: float = 1e-3,
    prewarm: int = 5,
):
    engine = Engine(
        model,
        inputs,
        model_name=model_name,
        device=device,
        DEBUG=False,
    )

    compiled = torch.compile(model)

    for _ in range(prewarm):
        model(*inputs)
        compiled(*inputs)
        engine.run(inputs)

    ok, info = check_correctness(
        engine,
        model,
        inputs,
        atol=correctness_atol,
        rtol=correctness_rtol,
    )

    assert ok, info

    print(f"\n=== {label} ===")

    pytorch_ms = benchmark(
        "PyTorch",
        lambda: model(*inputs),
        device=device,
        warmup=warmup,
        iters=iters,
    )

    compile_ms = benchmark(
        "torch.compile",
        lambda: compiled(*inputs),
        device=device,
        warmup=warmup,
        iters=iters,
    )

    fxfusion_ms = benchmark(
        "FXFusion",
        lambda: engine.run(inputs),
        device=device,
        warmup=warmup,
        iters=iters,
    )

    results = {
        "PyTorch": pytorch_ms,
        "torch.compile": compile_ms,
        "FXFusion": fxfusion_ms,
    }

    record_results(label, filename, device, results)

    return results


# -----------------------------------------------------------------------------
# MLP Benchmarks
# -----------------------------------------------------------------------------

def run_mlp_forward(
    label: str,
    depth: int,
    width: int,
    batch: int,
    filename: str,
    device: str = "cpu",
    warmup: int = 100,
    iters: int = 5000,
):
    x = torch.randn(batch, width, device=device)

    model = make_mlp(
        depth=depth,
        width=width,
        device=device,
    )

    return run_model_forward(
        label=label,
        model_name=f"mlp_{depth}x{width}_b{batch}",
        model=model,
        inputs=[x],
        filename=filename,
        device=device,
        warmup=warmup,
        iters=iters,
    )


# -----------------------------------------------------------------------------
# ResNet Benchmarks
# -----------------------------------------------------------------------------

def run_resnet_forward(
    version: int,
    label: str | None = None,
    filename: str | None = None,
    batch: int = 1,
    image_size: int = 224,
    x: torch.Tensor | None = None,
    device: str = "cpu",
    warmup: int = 20,
    iters: int = 100,
):
    version = int(version)
    model_name = f"resnet{version}"

    if x is None:
        x = torch.randn(
            batch,
            3,
            image_size,
            image_size,
            device=device,
        )
    else:
        x = x.to(device)

    if label is None:
        label = f"ResNet{version} ({x.shape[0]} x {x.shape[1]} x {x.shape[2]} x {x.shape[3]})"

    if filename is None:
        filename = model_name

    model = make_resnet(
        version=version,
        device=device,
    )

    return run_model_forward(
        label=label,
        model_name=model_name,
        model=model,
        inputs=[x],
        filename=filename,
        device=device,
        warmup=warmup,
        iters=iters,
    )

# -----------------------------------------------------------------------------
# GPT Benchmarks
# -----------------------------------------------------------------------------

def run_gpt_forward(
    label: str,
    model_name: str,
    model: nn.Module,
    tokens: torch.Tensor,
    max_seq_len: int,
    current_len: int,
    filename: str,
    device: str = "cpu",
    warmup: int = 10,
    iters: int = 100,
):
    mask_builder = StaticDecoderMaskBuilder(max_seq_len=max_seq_len).to(device)

    static_buffer = make_static_buffer(
        tokens,
        max_seq_len=max_seq_len,
        pad_idx=0,
    )

    mask = mask_builder(
        static_buffer,
        current_len=current_len,
        pad_idx=0,
    )

    return run_model_forward(
        label=label,
        model_name=model_name,
        model=model,
        inputs=[static_buffer, mask],
        filename=filename,
        device=device,
        warmup=warmup,
        iters=iters,
        correctness_atol=1e-5,
        correctness_rtol=1e-5,
    )


def run_gpt_decode(
    label: str,
    model_name: str,
    model: nn.Module,
    tokens: torch.Tensor,
    max_seq_len: int,
    filename: str,
    device: str = "cpu",
    warmup: int = 3,
    iters: int = 20,
):
    _, initial_len = tokens.shape

    mask_builder = StaticDecoderMaskBuilder(max_seq_len=max_seq_len).to(device)

    dummy_buffer = make_static_buffer(
        tokens,
        max_seq_len=max_seq_len,
        pad_idx=0,
    )

    dummy_mask = mask_builder(
        dummy_buffer,
        current_len=initial_len,
        pad_idx=0,
    )

    engine = Engine(
        model,
        [dummy_buffer, dummy_mask],
        model_name=model_name,
        device=device,
        DEBUG=False,
    )

    compiled = torch.compile(model)

    for _ in range(3):
        greedy_decode_static(
            model,
            tokens.clone(),
            mask_builder=mask_builder,
            max_seq_len=max_seq_len,
        )

        greedy_decode_static(
            compiled,
            tokens.clone(),
            mask_builder=mask_builder,
            max_seq_len=max_seq_len,
        )

        engine_decode_static(
            engine,
            tokens.clone(),
            mask_builder=mask_builder,
            max_seq_len=max_seq_len,
        )

    with torch.no_grad():
        expected_tokens = greedy_decode_static(
            model,
            tokens.clone(),
            mask_builder=mask_builder,
            max_seq_len=max_seq_len,
        )

        actual_tokens = engine_decode_static(
            engine,
            tokens.clone(),
            mask_builder=mask_builder,
            max_seq_len=max_seq_len,
        )

    ok, info = compare_outputs(
        actual_tokens,
        expected_tokens,
    )

    assert ok, info

    print(f"\n=== {label} ===")

    pytorch_ms = benchmark(
        "PyTorch",
        lambda: greedy_decode_static(
            model,
            tokens.clone(),
            mask_builder=mask_builder,
            max_seq_len=max_seq_len,
        ),
        device=device,
        warmup=warmup,
        iters=iters,
    )

    compile_ms = benchmark(
        "torch.compile",
        lambda: greedy_decode_static(
            compiled,
            tokens.clone(),
            mask_builder=mask_builder,
            max_seq_len=max_seq_len,
        ),
        device=device,
        warmup=warmup,
        iters=iters,
    )

    fxfusion_ms = benchmark(
        "FXFusion",
        lambda: engine_decode_static(
            engine,
            tokens.clone(),
            mask_builder=mask_builder,
            max_seq_len=max_seq_len,
        ),
        device=device,
        warmup=warmup,
        iters=iters,
    )

    results = {
        "PyTorch": pytorch_ms,
        "torch.compile": compile_ms,
        "FXFusion": fxfusion_ms,
    }

    record_results(label, filename, device, results)

    return results