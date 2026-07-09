import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "py"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import make_gpt, run_gpt_forward, run_gpt_decode


DEVICE = "cuda:0"

FORWARD_WARMUP = 20
FORWARD_ITERS = 300

DECODE_WARMUP = 5
DECODE_ITERS = 50


def main():
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    d_model = 128
    h = 4
    vocab_size = 2000
    expansion_factor = 4
    dropout = 0.0
    nx = 12

    batch_size = 1
    initial_len = 1
    max_seq_len = 32

    model = make_gpt(
        d_model=d_model,
        h=h,
        vocab_size=vocab_size,
        expansion_factor=expansion_factor,
        dropout=dropout,
        nx=nx,
        device=DEVICE,
    )

    tokens = torch.randint(
        1,
        vocab_size,
        (batch_size, initial_len),
        device=DEVICE,
    )

    run_gpt_forward(
        label="GPT dispatch-heavy forward static (Nx=12, d_model=128, B=1, T=32)",
        model_name="gpt_dispatch_nx12_forward_static",
        model=model,
        tokens=tokens,
        max_seq_len=max_seq_len,
        current_len=initial_len,
        filename="gpt_dispatch_nx12_forward_static",
        device=DEVICE,
        warmup=FORWARD_WARMUP,
        iters=FORWARD_ITERS,
    )

    run_gpt_decode(
        label="GPT dispatch-heavy static decode, full recompute (Nx=12, d_model=128, B=1, 1→32)",
        model_name="gpt_dispatch_nx12_static_decode",
        model=model,
        tokens=tokens,
        max_seq_len=max_seq_len,
        filename="gpt_dispatch_nx12_static_decode",
        device=DEVICE,
        warmup=DECODE_WARMUP,
        iters=DECODE_ITERS,
    )


if __name__ == "__main__":
    main()