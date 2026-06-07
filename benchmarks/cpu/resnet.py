import torch
import sys
from pathlib import Path
from torchvision.models import resnet18, resnet50

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "py"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fxfusion.engine import Engine
from utils import benchmark, save_chart, save_csv

DEVICE = "cpu"

def run(label: str, model_name: str, model, x: torch.Tensor, filename: str):
    engine = Engine(model, [x], model_name=model_name, device=DEVICE)
    compiled = torch.compile(model)

    for _ in range(5):
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

    run("ResNet18 (1 x 3 x 224 x 224)", "resnet18",
        resnet18(weights=None).eval().to(DEVICE),
        torch.randn(1, 3, 224, 224, device=DEVICE), "resnet18")

    run("ResNet50 (1 x 3 x 224 x 224)", "resnet50",
        resnet50(weights=None).eval().to(DEVICE),
        torch.randn(1, 3, 224, 224, device=DEVICE), "resnet50")


if __name__ == "__main__":
    main()