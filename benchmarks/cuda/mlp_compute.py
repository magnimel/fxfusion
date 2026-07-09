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
        label="Compute-bound MLP (depth=4, width=4096, batch=64)",
        depth=4,
        width=4096,
        batch=64,
        filename="mlp_compute_4x4096_b64",
        device=DEVICE,
        warmup=20,
        iters=200,
    )


if __name__ == "__main__":
    main()