import torch

def causal_mask(tokens: torch.Tensor) -> torch.Tensor:
    T = tokens.size(1)
    # causal_mask:  [seq_len, seq_len] 
    return torch.tril(torch.ones((T, T),
            device=tokens.device,
            dtype=torch.bool,
        )
    )

def padding_mask(tokens: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    # padding_mask: [batch, 1, 1, seq_len]
    return (tokens != pad_idx).unsqueeze(1).unsqueeze(2)


def decoder_mask(tokens: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    return padding_mask(tokens, pad_idx) & causal_mask(tokens)


def encoder_mask(tokens: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    return padding_mask(tokens, pad_idx)