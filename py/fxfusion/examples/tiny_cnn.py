from fxfusion.engine import Engine
import torch
import torch.nn as nn
from tests.utils import compare_outputs


DEVICE = "cpu"

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 63, kernel_size=3, stride=1, padding=1, groups=3),
            nn.ReLU(),
            nn.Conv2d(63, 4, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


@torch.inference_mode()
def main():

    model = TinyCNN().eval().to(DEVICE)
    x = torch.randn(1, 3, 224, 224,  device=DEVICE)
    out2 = model(x)

    engine = Engine(model, [x], model_name="tiny_cnn", device=DEVICE, DEBUG=True)
    out1 = engine.run([x])[0]
    
    ok, info = compare_outputs(out1, out2, rtol=1e-3, atol=1e-3)
    print("Outputs match:", ok)
    print("Info:", info)

if __name__ == "__main__":
    main()