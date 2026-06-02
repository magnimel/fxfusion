from fxfusion.engine import Engine
from utils import benchmark, check_correctness
import time
import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)

def main():
    torch.set_grad_enabled(False)

    model = TinyMLP().eval()
    x = torch.randn(64, 4096)

    engine = Engine(
        model,
        [x],
        model_name="tiny_mlp",
        device="cpu",
        DEBUG=True,
    )

    check_correctness(engine, model, x)

    compiled = torch.compile(model)

    pytorch_ms = benchmark("PyTorch", lambda: model(x))
    fxfusion_ms = benchmark("FXFusion", lambda: engine.run([x]))
    compile_ms = benchmark("torch.compile", lambda: compiled(x))

    print(f"\nSpeedup")
    print(f"FXFusion vs Pytorch: {pytorch_ms / fxfusion_ms:.2f}x")
    print(f"FXFusion vs compile: {compile_ms / fxfusion_ms:.2f}x")


if __name__ == "__main__":
    main()