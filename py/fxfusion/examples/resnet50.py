from fxfusion.engine import Engine
import torch
from torchvision.models import resnet50
from tests.utils import compare_outputs


@torch.inference_mode()
def main():
    model = resnet50(weights=None).eval()
    x = torch.randn(1, 3, 224, 224)
    out2 = model(x)

    engine = Engine(model, [x], model_name="resnet50", device="cpu", DEBUG=True)
    out1 = engine.run([x])[0]

    ok, info = compare_outputs(out1, out2, rtol=1e-3, atol=1e-3)
    print("Outputs match:", ok)
    print("Info:", info)


if __name__ == "__main__":
    main()