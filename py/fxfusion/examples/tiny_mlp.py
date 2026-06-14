from fxfusion.engine import Engine
import torch
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4096, elementwise_affine=False, bias=False)
        self.layers = nn.Sequential(
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(),
            nn.Linear(4096, 4096, bias=False),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.norm(x + self.layers(x))

    
@torch.inference_mode()
def main():
    
    model = TinyMLP().eval()
    x = torch.randn(1, 4096)
    engine = Engine(model, [x], model_name="tiny_mlp", device="cpu", DEBUG=True)
    out = engine.run(x)[0]
    print(out)

if __name__ == "__main__":
    main()