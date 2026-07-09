import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "py"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import make_gpt, run_gpt_forward, run_gpt_decode


DEVICE = "cuda:0"

FORWARD_WARMUP = 10
FORWARD_ITERS = 100

DECODE_WARMUP = 3
DECODE_ITERS = 20


def main():
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    d_model = 512
    h = 8
    vocab_size = 10000
    expansion_factor = 4
    dropout = 0.0
    nx = 12

    batch_size = 2
    initial_len = 5
    max_seq_len = 10

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
        label="GPT compute-heavy forward static (Nx=12, d_model=512, B=2, T=10)",
        model_name="gpt_compute_nx12_forward_static",
        model=model,
        tokens=tokens,
        max_seq_len=max_seq_len,
        current_len=initial_len,
        filename="gpt_compute_nx12_forward_static",
        device=DEVICE,
        warmup=FORWARD_WARMUP,
        iters=FORWARD_ITERS,
    )

    run_gpt_decode(
        label="GPT compute-heavy static decode, full recompute (Nx=12, d_model=512, B=2, 5→10)",
        model_name="gpt_compute_nx12_static_decode",
        model=model,
        tokens=tokens,
        max_seq_len=max_seq_len,
        filename="gpt_compute_nx12_static_decode",
        device=DEVICE,
        warmup=DECODE_WARMUP,
        iters=DECODE_ITERS,
    )


if __name__ == "__main__":
    main()