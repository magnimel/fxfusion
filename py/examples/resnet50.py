from torchvision.models import resnet50
from fxfusion.engine import Engine
from tests.utils import benchmark
import torch


def main():
    torch.set_grad_enabled(False)

    model = resnet50(weights=None).eval()
    x = torch.randn(1, 3, 224, 224)

    engine = Engine(model, [x], model_name="resnet50", device="cpu")
    compiled = torch.compile(model)

    print("ResNet50 (1 x 3 x 224 x 224)\n")
    pytorch_ms  = benchmark("PyTorch",       lambda: model(x))
    compile_ms  = benchmark("torch.compile", lambda: compiled(x))
    fxfusion_ms = benchmark("FXFusion",      lambda: engine.run([x]))

    print(f"\nFXFusion vs PyTorch      : {pytorch_ms  / fxfusion_ms:.2f}x")
    print(f"FXFusion vs torch.compile: {compile_ms  / fxfusion_ms:.2f}x")


if __name__ == "__main__":
    main()