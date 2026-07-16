import torch

from fxfusion.engine import Engine
from fxfusion.models.transformer.models.gpt import GPT
from fxfusion.models.transformer.layers.masks import (
    StaticDecoderMaskBuilder, make_static_buffer,
)
from fxfusion.models.transformer.inference import (
    greedy_decode_static, engine_decode_static,
)
from tests.utils import compare_outputs

DEVICE = "cpu"

@torch.inference_mode()
def main():
    d_model = 512
    h = 8
    vocab_size = 10000
    expansion_factor = 4
    dropout = 0.0
    Nx = 1

    batch_size = 1
    initial_len = 5
    max_seq_len = 10

    torch.manual_seed(6)  # fixed seed — reproduces the known-diverging case

    model = GPT(d_model, h, vocab_size, expansion_factor, dropout, Nx).eval().to(DEVICE)

    tokens = torch.randint(1, vocab_size, (batch_size, initial_len), device=DEVICE)
    mask_builder = StaticDecoderMaskBuilder(max_seq_len=max_seq_len).to(DEVICE)

    static_buffer = make_static_buffer(tokens, max_seq_len=max_seq_len, pad_idx=0)

    engine = Engine(
        model,
        [static_buffer, mask_builder(static_buffer, current_len=initial_len, pad_idx=0)],
        model_name="gpt",
        device=DEVICE,
        DEBUG=True,
    )

    torch_tokens = greedy_decode_static(
        model,
        tokens.clone(),
        mask_builder=mask_builder,
        max_seq_len=max_seq_len,
    )

    engine_tokens = engine_decode_static(
        engine,
        tokens.clone(),
        mask_builder=mask_builder,
        max_seq_len=max_seq_len,
    )

    ok, info = compare_outputs(engine_tokens, torch_tokens)
    print("Outputs match:", ok)
    print("Info:", info)


if __name__ == "__main__":
    main()