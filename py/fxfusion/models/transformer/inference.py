import torch

from fxfusion.engine import Engine
from fxfusion.models.transformer.layers.masks import StaticDecoderMaskBuilder, make_static_buffer

@torch.no_grad()
def greedy_decode_static(model, tokens: torch.Tensor, mask_builder: StaticDecoderMaskBuilder, max_seq_len: int = 50):
    batch_size, initial_len = tokens.shape

    static_buffer = make_static_buffer(tokens, max_seq_len)
    current_len = initial_len

    for _ in range(max_seq_len - initial_len):
        # Generate mask using the cached builder
        mask = mask_builder(static_buffer, current_len)

        logits = model(static_buffer, mask)

        next_token = logits[:, current_len - 1, :].argmax(dim=-1)

        static_buffer[:, current_len] = next_token
        current_len += 1

    return static_buffer[:, :current_len]


@torch.no_grad()
def engine_decode_static(engine: Engine, tokens: torch.Tensor, mask_builder: StaticDecoderMaskBuilder, max_seq_len: int = 50):
    batch_size, initial_len = tokens.shape

    static_buffer = make_static_buffer(tokens, max_seq_len)
    current_len = initial_len

    for _ in range(max_seq_len - initial_len):
        # Generate mask using the cached builder
        mask = mask_builder(static_buffer, current_len)

        logits = engine.run([static_buffer, mask])[0]

        next_token = logits[:, current_len - 1, :].argmax(dim=-1)

        static_buffer[:, current_len] = next_token
        current_len += 1

    return static_buffer[:, :current_len]

