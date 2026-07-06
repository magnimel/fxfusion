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

    model = GPT(d_model, h, vocab_size, expansion_factor, dropout, Nx).eval()

    tokens = torch.randint(1, vocab_size, (batch_size, initial_len))
    mask_builder = StaticDecoderMaskBuilder(max_seq_len=max_seq_len)

    static_tokens = make_static_buffer(tokens, max_seq_len=max_seq_len, pad_idx=0)
    static_mask = mask_builder(static_tokens, current_len=initial_len, pad_idx=0)

    engine = Engine(
        model,
        [static_tokens, static_mask],
        model_name="gpt",
        device="cpu",
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
    print(engine_tokens)


if __name__ == "__main__":
    main()