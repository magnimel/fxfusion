import torch
import torch.nn as nn
from layers.feedforward import FeedForward
from layers.attention import MultiHeadAttention   

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, h: int, expansion_factor: int, dropout: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, h, dropout)
        self.ff = FeedForward(d_model, expansion_factor)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) 
        
    def forward(self, x, mask=None):
        norm_x = self.norm1(x)
        x = x + self.dropout(self.mha(norm_x, norm_x, norm_x, mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x