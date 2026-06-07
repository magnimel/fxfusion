import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "py"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch.set_grad_enabled(False)

from mlp_dispatch import main as run_mlp_dispatch
from mlp_compute import main as run_mlp_compute
from resnet import main as run_resnet

if __name__ == "__main__":
    run_mlp_dispatch()
    run_mlp_compute()
    run_resnet()
