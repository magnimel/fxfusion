import torch
import torch.nn as nn

from fxfusion.passes.fusion_pass import FusionPass


def get_fused_model(model: nn.Module) -> nn.Module:
    return FusionPass().run(model)


def get_opcodes(model: nn.Module, print_graph: bool = False) -> list[str]:
    fused = get_fused_model(model)

    if print_graph:
        fused.graph.print_tabular()

    return [
        node.target.__name__
        for node in fused.graph.nodes
        if node.op == "call_function" and hasattr(node.target, "__name__")
    ]


def get_getattr_targets(model: nn.Module) -> list[str]:
    fused = get_fused_model(model)

    return [
        node.target
        for node in fused.graph.nodes
        if node.op == "get_attr"
    ]


def test_conv_bn_relu_fused():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.bn = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    opcodes = get_opcodes(M().eval())

    assert "conv2d_relu" in opcodes
    assert "conv2d" not in opcodes


def test_conv_bn_fused():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.bn = nn.BatchNorm2d(64)

        def forward(self, x):
            return self.bn(self.conv(x))

    opcodes = get_opcodes(M().eval())

    assert "conv2d" in opcodes


def test_conv_relu_fused():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv(x))

    opcodes = get_opcodes(M().eval())

    assert "conv2d_relu" in opcodes
    assert "conv2d" not in opcodes


def test_linear_relu_fused():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(128, 64)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.fc(x))

    opcodes = get_opcodes(M().eval())

    assert "linear_relu" in opcodes
    assert "linear" not in opcodes


def test_linear_fused():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(128, 64)

        def forward(self, x):
            return self.fc(x)

    opcodes = get_opcodes(M().eval())

    assert "linear" in opcodes


def test_add_relu_fused():
    class M(nn.Module):
        def forward(self, x):
            return torch.relu(x + x)

    opcodes = get_opcodes(M().eval())

    assert "add_relu" in opcodes


def test_no_fusion_across_branch():
    """Conv whose output is used by two nodes should not fuse with relu."""
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            y = self.conv(x)
            return self.relu(y) + y

    opcodes = get_opcodes(M().eval())

    assert "conv2d_relu" not in opcodes


def test_embedding_fused():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(1000, 64)

        def forward(self, x):
            return self.emb(x)

    opcodes = get_opcodes(M().eval())

    assert "embedding" in opcodes


def test_layernorm_fused():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(64)

        def forward(self, x):
            return self.norm(x)

    opcodes = get_opcodes(M().eval())

    assert "layernorm" in opcodes
    assert "add_layernorm" not in opcodes


def test_add_layernorm_fused():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(64)

        def forward(self, x):
            return self.norm(x + x)

    opcodes = get_opcodes(M().eval())

    assert "add_layernorm" in opcodes
    assert "add" not in opcodes
    assert "layernorm" not in opcodes


def test_add_layernorm_not_fused_across_branch():
    """Add whose output is reused should not be consumed by add_layernorm fusion."""
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(64)

        def forward(self, x):
            y = x + x
            return self.norm(y) + y

    opcodes = get_opcodes(M().eval())

    assert "add_layernorm" not in opcodes


def test_feedforward_fused():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 256)
            self.fc2 = nn.Linear(256, 64)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    opcodes = get_opcodes(M().eval())

    assert "feedforward" in opcodes
    assert "linear_relu" not in opcodes
    assert "linear" not in opcodes


def test_qkv_linear_fused():
    class M(nn.Module):
        def __init__(self):
            super().__init__()

            self.mha_w_q = nn.Linear(64, 64)
            self.mha_w_k = nn.Linear(64, 64)
            self.mha_w_v = nn.Linear(64, 64)

        def forward(self, x):
            B, T, C = x.shape

            q = self.mha_w_q(x)
            q = q.view(B, T, 8, 8)
            q = q.transpose(1, 2)

            k = self.mha_w_k(x)
            k = k.view(B, T, 8, 8)
            k = k.transpose(1, 2)

            v = self.mha_w_v(x)
            v = v.view(B, T, 8, 8)
            v = v.transpose(1, 2)

            return q + k + v

    opcodes = get_opcodes(M().eval())

    assert "qkv_linear" in opcodes
    assert opcodes.count("qkv_linear") == 1
    assert "linear" not in opcodes


def test_qkv_linear_packed_weights_created_correctly():
    class M(nn.Module):
        def __init__(self):
            super().__init__()

            self.mha_w_q = nn.Linear(64, 64)
            self.mha_w_k = nn.Linear(64, 64)
            self.mha_w_v = nn.Linear(64, 64)

        def forward(self, x):
            B, T, C = x.shape

            q = self.mha_w_q(x).view(B, T, 8, 8).transpose(1, 2)
            k = self.mha_w_k(x).view(B, T, 8, 8).transpose(1, 2)
            v = self.mha_w_v(x).view(B, T, 8, 8).transpose(1, 2)

            return q + k + v

    model = M().eval()
    fused = get_fused_model(model)

    get_attrs = [
        node.target
        for node in fused.graph.nodes
        if node.op == "get_attr"
    ]

    weight_attrs = [
        name
        for name in get_attrs
        if str(name).endswith("qkv_fused_weight")
    ]

    bias_attrs = [
        name
        for name in get_attrs
        if str(name).endswith("qkv_fused_bias")
    ]

    assert len(weight_attrs) == 1
    assert len(bias_attrs) == 1

    packed_weight = getattr(fused, weight_attrs[0])
    packed_bias = getattr(fused, bias_attrs[0])

    expected_weight = torch.cat(
        [
            model.mha_w_q.weight,
            model.mha_w_k.weight,
            model.mha_w_v.weight,
        ],
        dim=0,
    )

    expected_bias = torch.cat(
        [
            model.mha_w_q.bias,
            model.mha_w_k.bias,
            model.mha_w_v.bias,
        ],
        dim=0,
    )

    assert torch.equal(packed_weight, expected_weight)
    assert torch.equal(packed_bias, expected_bias)


def test_qkv_linear_not_fused_if_inputs_differ():
    class M(nn.Module):
        def __init__(self):
            super().__init__()

            self.mha_w_q = nn.Linear(64, 64)
            self.mha_w_k = nn.Linear(64, 64)
            self.mha_w_v = nn.Linear(64, 64)

        def forward(self, x, y):
            B, T, C = x.shape

            q = self.mha_w_q(x).view(B, T, 8, 8).transpose(1, 2)
            k = self.mha_w_k(y).view(B, T, 8, 8).transpose(1, 2)
            v = self.mha_w_v(x).view(B, T, 8, 8).transpose(1, 2)

            return q + k + v

    opcodes = get_opcodes(M().eval())

    assert "qkv_linear" not in opcodes


def test_inter_phase_dead_code_elimination():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 256)
            self.fc2 = nn.Linear(256, 64)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    fused = get_fused_model(M().eval())

    call_functions = [
        node
        for node in fused.graph.nodes
        if node.op == "call_function"
    ]

    assert len(call_functions) == 1
    assert call_functions[0].target.__name__ == "feedforward"
