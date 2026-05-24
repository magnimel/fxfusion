from torchvision.models import resnet18
from fxfusion.compiler import Compiler
import torch.fx as fx
import torch


def main():
    
    model = resnet18(weights=None)
    input = torch.randn(1, 3, 224, 224)
    fx_model: fx.GraphModule = Compiler().run(model, input)
            
if __name__ == "__main__":
    main()
    