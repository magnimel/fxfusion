import torch
import torch.nn as nn
import pytest
from fxfusion.engine import Engine
from tests.utils import check_correctness

torch.set_grad_enabled(False)


def run(name: str, model: nn.Module, x: torch.Tensor):
    engine = Engine(model, [x], model_name=name, device="cpu", DEBUG=False)
    ok, info = check_correctness(engine, model, x)
    assert ok, info


def test_conv2d():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        def forward(self, x): return self.conv(x)
    run("conv2d", M().eval(), torch.randn(1, 3, 224, 224))


def test_conv2d_relu():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
            self.relu = nn.ReLU()
        def forward(self, x): return self.relu(self.conv(x))
    run("conv2d_relu", M().eval(), torch.randn(1, 3, 224, 224))


def test_max_pool2d():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        def forward(self, x): return self.pool(x)
    run("max_pool2d", M().eval(), torch.randn(1, 64, 112, 112))


def test_avg_pool2d():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        def forward(self, x): return self.pool(x)
    run("avg_pool2d", M().eval(), torch.randn(1, 64, 112, 112))


def test_adaptive_avg_pool2d():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        def forward(self, x): return self.pool(x)
    run("adaptive_avg_pool2d", M().eval(), torch.randn(1, 512, 7, 7))


def test_linear():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(512, 1000)
        def forward(self, x): return self.fc(x)
    run("linear", M().eval(), torch.randn(1, 512))


def test_linear_relu():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(512, 256)
            self.relu = nn.ReLU()
        def forward(self, x): return self.relu(self.fc(x))
    run("linear_relu", M().eval(), torch.randn(1, 512))


def test_add_relu():
    class M(nn.Module):
        def forward(self, x): return torch.relu(x + x)
    run("add_relu", M().eval(), torch.randn(1, 64, 56, 56))


def test_residual_block():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        def forward(self, x):
            return torch.relu(self.conv2(torch.relu(self.conv1(x))) + x)
    run("residual_block", M().eval(), torch.randn(1, 64, 56, 56))


def test_residual_block_downsample():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
            self.downsample = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)
        def forward(self, x):
            return torch.relu(self.conv2(torch.relu(self.conv1(x))) + self.downsample(x))
    run("residual_block_downsample", M().eval(), torch.randn(1, 64, 56, 56))


def test_mlp():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 10),
            )
        def forward(self, x): return self.layers(x)
    run("mlp", M().eval(), torch.randn(4, 256))


def test_resnet18():
    from torchvision.models import resnet18
    run("resnet18", resnet18(weights=None).eval(), torch.randn(1, 3, 224, 224))