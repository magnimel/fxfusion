import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "py"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import run_mlp_forward


DEVICE = "cuda:0"


def main():
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    run_mlp_forward(
        label="Dispatch-heavy MLP (depth=32, width=64, batch=1)",
        depth=32,
        width=64,
        batch=1,
        filename="mlp_dispatch_32x64_b1",
        device=DEVICE,
        warmup=100,
        iters=5000,
    )

    run_mlp_forward(
        label="Dispatch-heavy MLP (depth=32, width=128, batch=1)",
        depth=32,
        width=128,
        batch=1,
        filename="mlp_dispatch_32x128_b1",
        device=DEVICE,
        warmup=100,
        iters=5000,
    )

    run_mlp_forward(
        label="Balanced MLP (depth=16, width=256, batch=4)",
        depth=16,
        width=256,
        batch=4,
        filename="mlp_balanced_16x256_b4",
        device=DEVICE,
        warmup=100,
        iters=3000,
    )


if __name__ == "__main__":
    main()