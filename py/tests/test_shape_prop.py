import torch
import torch.nn as nn
import torch.fx as fx
import pytest

from fxfusion.passes.fusion_pass import FusionPass
from fxfusion.passes.shape_prop import ShapePropPass


def propagate(model: nn.Module, x: torch.Tensor) -> fx.GraphModule:
    fused = FusionPass().run(model)
    ShapePropPass(fused).propagate(x)
    return fused


def shapes(fused: fx.GraphModule) -> dict[str, tuple]:
    return {
        node.name: node.meta["shape"]
        for node in fused.graph.nodes
        if "shape" in node.meta
    }


def check_output_shape(model: nn.Module, x: torch.Tensor):
    """Output shape of last non-output node must match PyTorch."""
    fused = propagate(model, x)
    expected = tuple(model(x).shape)
    # find the node the output node points to
    for node in fused.graph.nodes:
        if node.op == "output":
            src = node.args[0]
            if isinstance(src, fx.Node):
                assert fused.graph.nodes
                actual = src.meta.get("shape")
                assert actual == expected, f"Shape mismatch: got {actual}, expected {expected}"
            return
    pytest.fail("No output node found")


# -----------------------------------------------------------------------------
# Shape correctness per op
# -----------------------------------------------------------------------------

def test_shape_conv2d():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        def forward(self, x): return self.conv(x)
    check_output_shape(M().eval(), torch.randn(1, 3, 224, 224))


def test_shape_conv2d_relu():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, stride=2, padding=1)
            self.relu = nn.ReLU()
        def forward(self, x): return self.relu(self.conv(x))
    check_output_shape(M().eval(), torch.randn(1, 3, 224, 224))


def test_shape_linear():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(512, 1000)
        def forward(self, x): return self.fc(x)
    check_output_shape(M().eval(), torch.randn(4, 512))


def test_shape_linear_relu():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc   = nn.Linear(512, 256)
            self.relu = nn.ReLU()
        def forward(self, x): return self.relu(self.fc(x))
    check_output_shape(M().eval(), torch.randn(4, 512))


def test_shape_max_pool2d():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        def forward(self, x): return self.pool(x)
    check_output_shape(M().eval(), torch.randn(1, 64, 112, 112))


def test_shape_avg_pool2d():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool2d(3, stride=2, padding=1)
        def forward(self, x): return self.pool(x)
    check_output_shape(M().eval(), torch.randn(1, 64, 112, 112))


def test_shape_adaptive_avg_pool2d():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        def forward(self, x): return self.pool(x)
    check_output_shape(M().eval(), torch.randn(1, 512, 7, 7))


def test_shape_add_relu():
    class M(nn.Module):
        def forward(self, x): return torch.relu(x + x)
    check_output_shape(M().eval(), torch.randn(1, 64, 56, 56))


def test_shape_flatten():
    class M(nn.Module):
        def forward(self, x): return torch.flatten(x, 1)
    check_output_shape(M().eval(), torch.randn(1, 512, 7, 7))


def test_shape_residual_block():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        def forward(self, x):
            return torch.relu(self.conv2(torch.relu(self.conv1(x))) + x)
    check_output_shape(M().eval(), torch.randn(1, 64, 56, 56))


def test_shape_resnet18():
    from torchvision.models import resnet18
    check_output_shape(resnet18(weights=None).eval(), torch.randn(1, 3, 224, 224))


# -----------------------------------------------------------------------------
# Meta fields
# -----------------------------------------------------------------------------

def test_all_nodes_have_shape():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc   = nn.Linear(64, 10)
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            return self.fc(x.flatten(1))

    fused = propagate(M().eval(), torch.randn(1, 3, 8, 8))
    for node in fused.graph.nodes:
        if node.op in ("output", "get_attr"):
            continue
        assert "shape" in node.meta, f"Missing shape on node: {node.name}"
        assert "dtype" in node.meta, f"Missing dtype on node: {node.name}"

def test_dtype_is_float32():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
        def forward(self, x): return self.conv(x)

    fused = propagate(M().eval(), torch.randn(1, 3, 32, 32))
    for node in fused.graph.nodes:
        if "dtype" in node.meta:
            assert node.meta["dtype"] == torch.float32, \
                f"Expected float32 on {node.name}, got {node.meta['dtype']}"


def test_shapes_consistent_with_memory_plan():
    """Shape sizes must match the size_bytes computed by the memory planner."""
    from fxfusion.passes.memory_plan import MemoryPlanningPass

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        def forward(self, x): return self.conv2(self.conv1(x))

    fused = propagate(M().eval(), torch.randn(1, 3, 32, 32))
    MemoryPlanningPass(fused).run()

    for node in fused.graph.nodes:
        alloc = node.meta.get("alloc")
        shape = node.meta.get("shape")
        dtype = node.meta.get("dtype")

        if alloc is None or shape is None or dtype is None:
            continue

        if alloc.kind not in ("activation",):
            continue

        expected_bytes = (
            torch.empty(shape, dtype=dtype).numel()
            * torch.empty([], dtype=dtype).element_size()
        )

        # allow for alignment padding
        assert alloc.size_bytes >= expected_bytes, \
            f"{node.name}: size_bytes {alloc.size_bytes} < expected {expected_bytes}"