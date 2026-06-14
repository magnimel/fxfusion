from torchvision.models import resnet50
from fxfusion.engine import Engine
import torch

@torch.inference_mode()
def main():
    
    model = resnet50(weights=None).eval()
    x = torch.randn(1, 3, 224, 224)
    engine = Engine(model, [x], model_name="resnet50", device="cpu", DEBUG=True)
    out = engine.run(x)[0]
    print(out)
    

if __name__ == "__main__":
    main()