import torch
import torch.nn as nn

# Mask convention used by the model:
#   True  = keep / allowed
#   False = masked / blocked
#
# This matches:
#   scores = scores.masked_fill(mask == 0, -inf)


def none_mask(tokens: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    # none_mask: [batch, 1, seq_len, seq_len]
    # Allows every position.
    B, T = tokens.size(0), tokens.size(1)

    return torch.ones(
        (B, 1, T, T),
        device=tokens.device,
        dtype=torch.bool,
    )
    
def random_mask(tokens: torch.Tensor, p_keep: float = 0.5, seed: int | None = None) -> torch.Tensor:
    # random_mask: [batch, 1, seq_len, seq_len]
    # Randomly allows/blocks positions independently, useful for stress-testing
    # edge cases like fully-masked rows (all False for some query position),
    # which torch.softmax does NOT guard against — see note above.
    B, T = tokens.size(0), tokens.size(1)

    generator = torch.Generator(device=tokens.device)
    if seed is not None:
        generator.manual_seed(seed)

    return (
        torch.rand((B, 1, T, T), device=tokens.device, generator=generator) < p_keep
    )

def causal_mask(tokens: torch.Tensor) -> torch.Tensor:
    # causal_mask: [seq_len, seq_len]
    # Allows current and previous positions only.
    T = tokens.size(1)

    return torch.tril(
        torch.ones(
            (T, T),
            device=tokens.device,
            dtype=torch.bool,
        )
    )


def padding_mask(tokens: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    # padding_mask: [batch, 1, 1, seq_len]
    # Allows non-padding tokens.
    return (tokens != pad_idx).unsqueeze(1).unsqueeze(2)


def decoder_mask(tokens: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    # decoder_mask: [batch, 1, seq_len, seq_len]
    # Blocks padding and future positions.
    return padding_mask(tokens, pad_idx) & causal_mask(tokens)


def encoder_mask(tokens: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    # encoder_mask: [batch, 1, 1, seq_len]
    # Blocks padding positions.
    return padding_mask(tokens, pad_idx)


def make_static_buffer(
    tokens: torch.Tensor,
    max_seq_len: int,
    pad_idx: int = 0,
) -> torch.Tensor:
    # static_buffer: [batch, max_seq_len]
    # Stores generated tokens in a fixed-size buffer.
    B, T = tokens.shape

    if T > max_seq_len:
        raise ValueError(f"Token length {T} exceeds max_seq_len {max_seq_len}")

    static_buffer = torch.full(
        (B, max_seq_len),
        pad_idx,
        dtype=tokens.dtype,
        device=tokens.device,
    )

    static_buffer[:, :T] = tokens

    return static_buffer


def static_prefix_mask(tokens: torch.Tensor, current_len: int) -> torch.Tensor:
    # static_prefix_mask: [batch, 1, 1, max_seq_len]
    # Allows only positions before current_len.
    #
    # True  = filled static-buffer positions
    # False = unused static-buffer positions
    B, T = tokens.shape

    if current_len > T:
        raise ValueError(f"current_len {current_len} exceeds static length {T}")
        
    mask = torch.arange(T, device=tokens.device) < current_len
    return mask.view(1, 1, 1, T).expand(B, 1, 1, T)


def static_encoder_mask(tokens: torch.Tensor, current_len: int, pad_idx: int = 0) -> torch.Tensor:
    # static_encoder_mask: [batch, 1, 1, max_seq_len]
    #
    # Blocks:
    #   1. unused static-buffer positions after current_len
    #   2. padding positions
    return static_prefix_mask(tokens, current_len) & padding_mask(tokens, pad_idx)


def static_decoder_mask(tokens: torch.Tensor, current_len: int, pad_idx: int = 0) -> torch.Tensor:
    # static_decoder_mask: [batch, 1, max_seq_len, max_seq_len]
    #
    # Blocks:
    #   1. future positions through causal masking
    #   2. unused static-buffer key positions after current_len
    #   3. padding key positions
    return static_encoder_mask(tokens, current_len, pad_idx) & causal_mask(tokens)


class StaticDecoderMaskBuilder(nn.Module):
    """
    A stateful mask builder that caches the causal mask for the 
    maximum sequence length to avoid re-allocating memory during generation.
    """
    def __init__(self, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        # Pre-compute the absolute maximum causal mask once.
        causal_matrix = torch.tril(torch.ones((max_seq_len, max_seq_len), dtype=torch.bool))
        self.register_buffer("cached_causal_mask", causal_matrix)

    def forward(self, tokens: torch.Tensor, current_len: int, pad_idx: int = 0) -> torch.Tensor:
        """
        static_decoder_mask: [batch, 1, max_seq_len, max_seq_len]
        """
        B, T = tokens.shape
        if T > self.max_seq_len:
            raise ValueError(f"Token length {T} exceeds max_seq_len {self.max_seq_len}")
        
        if self.cached_causal_mask.device != tokens.device:
            raise RuntimeError(f"Device mismatch: call mask_builder.to({tokens.device}) before use")
        
        prefix_padding_mask = static_prefix_mask(tokens, current_len) & padding_mask(tokens, pad_idx)
        sliced_causal = self.cached_causal_mask[:T, :T]
        return prefix_padding_mask & sliced_causal