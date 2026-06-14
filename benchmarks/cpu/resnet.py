import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "py"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import run_resnet_forward


DEVICE = "cpu"


def main():
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    run_resnet_forward(
        version=18,
        device=DEVICE,
        batch=1,
        warmup=20,
        iters=100,
    )

    run_resnet_forward(
        version=50,
        device=DEVICE,
        batch=1,
        warmup=10,
        iters=50,
    )



if __name__ == "__main__":
    main()