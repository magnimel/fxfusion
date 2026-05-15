from torchvision.models import resnet18
from ..fxfusion.passes.fusion_pass import FXFusionPass
from ..fxfusion.passes.shape_prop import FXFusionShapeProp
import torch.fx as fx
import torch


def main():
    
    model = resnet18(weights=None)
    print(model)

    FXFusionPass().run(model)

    fx_model = fx.symbolic_trace(model)
    fx_model.graph.print_tabular()    
    
    input = torch.randn(1, 3, 224, 224)
    FXFusionShapeProp(fx_model).propagate(input)
    
    for node in fx_model.graph.nodes:
        print(node.name, " : ",node.meta["shape"], " | ", node.meta["dtype"], " | ", node.meta["device"], '\n')
            
if __name__ == "__main__":
    main()
    