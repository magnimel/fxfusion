import torch
import torch.nn as nn
from fxfusion.models.transformer.layers.embedding import InputEmbedding, PositionalEncoding
from fxfusion.models.transformer.blocks.gpt_block import GPTBlock


class GPT(nn.Module):
    def __init__(self, d_model: int, h: int, vocab_size: int, expansion_factor: int, dropout: float = 0.1, Nx: int = 12):
        super().__init__()
        assert Nx > 0
        self.norm = nn.LayerNorm(d_model)
        self.in_embedding = InputEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.gpt_blocks = nn.ModuleList([
            GPTBlock(d_model, h, expansion_factor, dropout)
            for _ in range(Nx)
        ])
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask):
        x = self.in_embedding(x)
        x = self.pos_encoding(x)

        for block in self.gpt_blocks:
            x = block(x, mask)

        x = self.norm(x)
        return self.linear(x)
