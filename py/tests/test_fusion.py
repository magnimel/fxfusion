import torch
import torch.nn as nn
import torch.fx as fx
from fxfusion.passes.fusion import Fusion
from fxfusion.passes.fusion_pass import FusionPass

def get_opcodes(model: nn.Module, x: torch.Tensor) -> list[str]:
    fused = FusionPass().run(model)
    return [
        node.target.__name__
        for node in fused.graph.nodes
        if node.op == "call_function" and hasattr(node.target, "__name__")
    ]

def test_conv_bn_relu_fused():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.bn   = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
        def forward(self, x): return self.relu(self.bn(self.conv(x)))

    opcodes = get_opcodes(M().eval(), torch.randn(1, 3, 32, 32))
    assert "fused_conv2d_relu" in opcodes
    assert "fused_conv2d" not in opcodes

def test_conv_bn_fused():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.bn   = nn.BatchNorm2d(64)
        def forward(self, x): return self.bn(self.conv(x))

    opcodes = get_opcodes(M().eval(), torch.randn(1, 3, 32, 32))
    assert "fused_conv2d" in opcodes

def test_linear_relu_fused():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc   = nn.Linear(128, 64)
            self.relu = nn.ReLU()
        def forward(self, x): return self.relu(self.fc(x))

    opcodes = get_opcodes(M().eval(), torch.randn(1, 128))
    assert "fused_linear_relu" in opcodes

def test_linear_fused():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(128, 64)
        def forward(self, x): return self.fc(x)

    opcodes = get_opcodes(M().eval(), torch.randn(1, 128))
    assert "fused_linear" in opcodes

def test_add_relu_fused():
    class M(nn.Module):
        def forward(self, x): return torch.relu(x + x)

    opcodes = get_opcodes(M().eval(), torch.randn(1, 64))
    assert "fused_add_relu" in opcodes

def test_no_fusion_across_branch():
    """Conv whose output is used by two nodes should not fuse with relu."""
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.relu = nn.ReLU()
        def forward(self, x):
            y = self.conv(x)
            return self.relu(y) + y  # y has two users

    opcodes = get_opcodes(M().eval(), torch.randn(1, 3, 32, 32))
    assert "fused_conv2d_relu" not in opcodes