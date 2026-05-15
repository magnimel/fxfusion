from fxfusion.passes.fusion_pass import FXFusionPass
from fxfusion.passes.shape_prop import FXFusionShapeProp
import torch.fx as fx
import torch.nn as nn
import torch


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)
    
    
def main():
    
    model = TinyMLP()
    print(model)

    fx_model : fx.GraphModule = FXFusionPass().run(model)
    fx_model.graph.print_tabular()    
    
    input = torch.randn(1, 50)
    FXFusionShapeProp(fx_model).propagate(input)
    
    for node in fx_model.graph.nodes:
        print(f"{node.name} : {node.meta['shape']} | {node.meta['dtype']} | {node.meta['device']}")
            
if __name__ == "__main__":
    main()
    