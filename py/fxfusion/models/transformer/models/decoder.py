import torch
import torch.nn as nn
from layers.embedding import InputEmbedding, PositionalEncoding
from blocks.decoder_block import DecoderBlock

class Decoder(nn.Module):
    def __init__(self, d_model: int, h: int, vocab_size: int, expansion_factor: int, dropout: float = 0.1, Nx: int = 1):
        super().__init__()
        assert Nx > 0
        self.norm = nn.LayerNorm(d_model)
        self.in_embedding = InputEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, h, expansion_factor, dropout)
            for _ in range(Nx)
        ])
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.in_embedding(x)
        x = self.pos_encoding(x)

        for block in self.decoder_blocks:
            x = block(x, encoder_output, src_mask, tgt_mask)
        x = self.norm(x)
        
        return self.linear(x)