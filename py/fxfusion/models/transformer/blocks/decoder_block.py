import torch
import torch.nn as nn
from layers.feedforward import FeedForward
from layers.attention import MultiHeadAttention   

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, h: int, expansion_factor: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.cross_attn = MultiHeadAttention(d_model, h, dropout)
        self.ff = FeedForward(d_model, expansion_factor)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        norm_x = self.norm1(x)
        x = x + self.dropout(self.self_attn(norm_x, norm_x, norm_x, tgt_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), encoder_output, encoder_output, src_mask))
        x = x + self.dropout(self.ff(self.norm3(x)))
        return x