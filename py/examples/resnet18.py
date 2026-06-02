from torchvision.models import resnet18
from fxfusion.engine import Engine
import torch.fx as fx
import torch
from utils import benchmark, check_correctness

def main():
    torch.set_grad_enabled(False)
    
    model = resnet18(weights=None).eval()
    
    x = torch.randn(1, 3, 224, 224)
    
    engine = Engine(
        model,
        [x],
        model_name="resnet18",
        device="cpu",
        DEBUG=True
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
    