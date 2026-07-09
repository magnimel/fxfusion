import torch
import torch.nn as nn
import pytest

from fxfusion.engine import Engine
from tests.utils import check_correctness, compare_outputs

from fxfusion.models.transformer.layers.masks import (
    StaticDecoderMaskBuilder,
    make_static_buffer,
)
from fxfusion.models.transformer.inference import (
    greedy_decode_static,
    engine_decode_static,
)
from fxfusion.models.transformer.models.gpt import GPT

torch.set_grad_enabled(False)


def _available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        print(f"[test_ops] CUDA available: {torch.cuda.get_device_name(0)} — running cpu and cuda:0 tests")
        devices.append("cuda:0")
    else:
        print("[test_ops] CUDA not available — running cpu tests only")
    return devices


@pytest.fixture(params=_available_devices())
def device(request):
    return request.param


def run(
    device: str,
    name: str,
    model: nn.Module,
    inputs: list[torch.Tensor],
    atol: float = 1e-3,
    rtol: float = 1e-3,
):
    model = model.to(device)
    inputs = [t.to(device) for t in inputs]

    engine = Engine(model, inputs, model_name=f"{name}_{device.replace(':', '')}", device=device, DEBUG=False)
    ok, info = check_correctness(engine, model, inputs, atol=atol, rtol=rtol)
    assert ok, info


def test_conv2d(device):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            return self.conv(x)

    run(device, "conv2d", M().eval(), [torch.randn(1, 3, 224, 224)])


def test_conv2d_relu(device):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv(x))

    run(device, "conv2d_relu", M().eval(), [torch.randn(1, 3, 224, 224)])


def test_max_pool2d(device):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        def forward(self, x):
            return self.pool(x)

    run(device, "max_pool2d", M().eval(), [torch.randn(1, 64, 112, 112)])


def test_avg_pool2d(device):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        def forward(self, x):
            return self.pool(x)

    run(device, "avg_pool2d", M().eval(), [torch.randn(1, 64, 112, 112)])


def test_adaptive_avg_pool2d(device):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            return self.pool(x)

    run(device, "adaptive_avg_pool2d", M().eval(), [torch.randn(1, 512, 7, 7)])


def test_linear(device):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(512, 1000)

        def forward(self, x):
            return self.fc(x)

    run(device, "linear", M().eval(), [torch.randn(1, 512)])


def test_linear_relu(device):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(512, 256)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.fc(x))

    run(device, "linear_relu", M().eval(), [torch.randn(1, 512)])


def test_add_relu(device):
    class M(nn.Module):
        def forward(self, x):
            return torch.relu(x + x)

    run(device, "add_relu", M().eval(), [torch.randn(1, 64, 56, 56)])


def test_embedding(device):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 64)

        def forward(self, x):
            return self.embedding(x)

    run(device, "embedding", M().eval(), [torch.randint(1, 1000, (2, 10))])


def test_layernorm(device):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(64)

        def forward(self, x):
            return self.norm(x)

    run(device, "layernorm", M().eval(), [torch.randn(2, 10, 64)])


def test_add_layernorm(device):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(64)

        def forward(self, x):
            return self.norm(x + x)

    run(device, "add_layernorm", M().eval(), [torch.randn(2, 10, 64)])


def test_feedforward(device):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 256)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(256, 64)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    run(device, "feedforward", M().eval(), [torch.randn(2, 10, 64)])


def test_residual_block(device):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            return torch.relu(self.conv2(torch.relu(self.conv1(x))) + x)

    run(device, "residual_block", M().eval(), [torch.randn(1, 64, 56, 56)])


def test_residual_block_downsample(device):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
            self.downsample = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)

        def forward(self, x):
            return torch.relu(self.conv2(torch.relu(self.conv1(x))) + self.downsample(x))

    run(device, "residual_block_downsample", M().eval(), [torch.randn(1, 64, 56, 56)])


def test_mlp(device):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 10),
            )

        def forward(self, x):
            return self.layers(x)

    run(device, "mlp", M().eval(), [torch.randn(4, 256)])


def test_transpose_multidim(device):
    class M(nn.Module):
        def forward(self, x):
            # transpose(0, 2): non-adjacent dims
            return x.transpose(0, 2)

    run(device, "transpose_multidim", M().eval(), [torch.randn(2, 5, 7, 3)])


def test_transpose_then_view(device):
    class M(nn.Module):
        def forward(self, x):
            h = x.transpose(0, 2)          # (2,8,6) -> (6,8,2)
            h = h.contiguous().view(6, 16) # (6,8,2) -> (6,16)
            return h.transpose(0, 1)       # (6,16) -> (16,6)

    run(device, "transpose_then_view", M().eval(), [torch.randn(2, 8, 6)])


def test_mul_tensor_tensor(device):
    class M(nn.Module):
        def forward(self, x):
            return x * x

    run(device, "mul_tensor_tensor", M().eval(), [torch.randn(4, 64)])


def test_mul_tensor_scalar_float(device):
    class M(nn.Module):
        def forward(self, x):
            a = x * 2.5
            return 0.5 * a

    run(device, "mul_tensor_scalar_float", M().eval(), [torch.randn(4, 64)])


def test_mul_tensor_scalar_int(device):
    class M(nn.Module):
        def forward(self, x):
            a = x * 3
            return 2 * a

    run(device, "mul_tensor_scalar_int", M().eval(), [torch.randn(4, 64)])


def test_gpt_forward_static_shape(device):
    torch.manual_seed(0)

    d_model = 32
    h = 4
    vocab_size = 128
    expansion_factor = 2
    dropout = 0.0
    Nx = 1

    batch_size = 2
    max_seq_len = 8
    initial_len = 5

    model = GPT(d_model, h, vocab_size, expansion_factor, dropout, Nx).eval()

    tokens = torch.randint(1, vocab_size, (batch_size, initial_len), device=device)

    static_buffer = make_static_buffer(tokens, max_seq_len=max_seq_len, pad_idx=0)

    mask_builder = StaticDecoderMaskBuilder(max_seq_len=max_seq_len).to(device)

    mask = mask_builder(static_buffer, current_len=initial_len, pad_idx=0)

    run(
        device,
        "gpt_forward_static_shape",
        model,
        [static_buffer, mask],
        atol=1e-5,
        rtol=1e-5,
    )


def test_gpt_static_decode(device):
    torch.manual_seed(0)

    d_model = 32
    h = 4
    vocab_size = 128
    expansion_factor = 2
    dropout = 0.0
    Nx = 1

    max_seq_len = 8
    initial_len = 5
    batch_size = 2

    model = GPT(d_model, h, vocab_size, expansion_factor, dropout, Nx).eval().to(device)

    input_tokens = torch.randint(1, vocab_size, (batch_size, initial_len), device=device)

    mask_builder = StaticDecoderMaskBuilder(max_seq_len=max_seq_len).to(device)

    expected_tokens = greedy_decode_static(
        model=model,
        tokens=input_tokens.clone(),
        mask_builder=mask_builder,
        max_seq_len=max_seq_len,
    )

    dummy_buffer = make_static_buffer(input_tokens, max_seq_len=max_seq_len, pad_idx=0)

    dummy_mask = mask_builder(dummy_buffer, current_len=initial_len, pad_idx=0)

    engine = Engine(
        model,
        [dummy_buffer, dummy_mask],
        model_name=f"gpt_static_decode_{device.replace(':', '')}",
        device=device,
        DEBUG=False,
    )

    actual_tokens = engine_decode_static(
        engine=engine,
        tokens=input_tokens.clone(),
        mask_builder=mask_builder,
        max_seq_len=max_seq_len,
    )

    ok, info = compare_outputs(actual_tokens, expected_tokens)
    assert ok, info


def test_mha_fully_masked_row_nan_parity(device):
    torch.manual_seed(0)

    d_model = 32
    h = 4
    vocab_size = 128
    expansion_factor = 2
    dropout = 0.0
    Nx = 1
    batch_size = 1
    seq_len = 6

    model = GPT(d_model, h, vocab_size, expansion_factor, dropout, Nx).eval()
    tokens = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)

    mask = torch.ones((batch_size, 1, seq_len, seq_len), dtype=torch.bool, device=device)
    mask[0, 0, 0, :] = False

    run(device, "mha_fully_masked_row", model, [tokens, mask], atol=1e-3, rtol=1e-3)


def test_resnet18(device):
    from torchvision.models import resnet18

    run(device, "resnet18", resnet18(weights=None).eval(), [torch.randn(1, 3, 224, 224)])