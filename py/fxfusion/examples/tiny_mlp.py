from fxfusion.engine import Engine
import torch
import torch.nn as nn
from tests.utils import compare_outputs


DEVICE = "cpu"

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4096, elementwise_affine=False, bias=False)
        self.layers = nn.Sequential(
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(),
            nn.Linear(4096, 4096, bias=False),
            nn.ReLU(),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(),
            nn.Linear(4096, 4096, bias=True)
        )

    def forward(self, x):
        x = self.norm(x + self.layers(x))
        return x.transpose(0, 1)


@torch.inference_mode()
def main():

    model = TinyMLP().eval().to(DEVICE)
    x = torch.randn(1, 4096, device=DEVICE)
    out2 = model(x)

    engine = Engine(model, [x], model_name="tiny_mlp", device=DEVICE, DEBUG=True)
    out1 = engine.run([x])[0]
    
    ok, info = compare_outputs(out1, out2, rtol=1e-3, atol=1e-3)
    print("Outputs match:", ok)
    print("Info:", info)

if __name__ == "__main__":
    main()