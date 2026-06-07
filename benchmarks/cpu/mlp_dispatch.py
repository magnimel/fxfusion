import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "py"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fxfusion.engine import Engine
from utils import benchmark, save_chart, save_csv, make_mlp

DEVICE = "cpu"

def run(label: str, depth: int, width: int, batch: int, filename: str):
    x = torch.randn(batch, width, device=DEVICE)
    model = make_mlp(depth, width, device=DEVICE)

    engine = Engine(model, [x], model_name=f"mlp_{depth}x{width}", device=DEVICE)
    compiled = torch.compile(model)

    for _ in range(10):
        compiled(x)

    print(f"\n=== {label} ===")
    pytorch_ms = benchmark("PyTorch", lambda: model(x), device=DEVICE)
    compile_ms = benchmark("torch.compile", lambda: compiled(x), device=DEVICE)
    fxfusion_ms = benchmark("FXFusion", lambda: engine.run([x]), device=DEVICE)

    results = {
        "PyTorch": pytorch_ms,
        "torch.compile": compile_ms,
        "FXFusion": fxfusion_ms,
    }

    save_chart(label, results, f"cpu_{filename}")
    save_csv(f"cpu_{filename}", label, DEVICE, results)


def main():
    torch.set_grad_enabled(False)

    run("Dispatch-heavy MLP (depth=32, width=64, batch=1)",
        depth=32, width=64, batch=1, filename="mlp_dispatch_32x64")

    run("Dispatch-heavy MLP (depth=32, width=128, batch=1)",
        depth=32, width=128, batch=1, filename="mlp_dispatch_32x128")

    run("Balanced MLP (depth=16, width=256, batch=4)",
        depth=16, width=256, batch=4, filename="mlp_balanced_16x256")


if __name__ == "__main__":
    main()