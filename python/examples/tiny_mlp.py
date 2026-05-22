from fxfusion.passes.fusion_pass import FusionPass
from fxfusion.compiler import FXFusionCompiler
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
    input = torch.randn(1, 50)
    fx_model: fx.GraphModule = FXFusionCompiler().run(model, input)
    
if __name__ == "__main__":
    main()
    