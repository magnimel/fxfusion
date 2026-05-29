from torchvision.models import resnet18
from fxfusion.engine import Engine
import torch.fx as fx
import torch


def main():
    
    model = resnet18(weights=None).eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    engine = Engine(
        model,
        [dummy_input],
        model_name="resnet18",
        device="cpu",
        DEBUG=False
    )

    outputs = engine.run([dummy_input])
    print(outputs)
            
if __name__ == "__main__":
    main()
    