from torchvision.models import resnet18
from fxfusion.compiler import FXFusionCompiler
import torch.fx as fx
import torch


def main():
    
    model = resnet18(weights=None)
    print(model)
        
    input = torch.randn(1, 3, 224, 224)
    fx_model: fx.GraphModule = FXFusionCompiler().run(model, input)
    
    fx_model.graph.print_tabular()    
    
            
if __name__ == "__main__":
    main()
    