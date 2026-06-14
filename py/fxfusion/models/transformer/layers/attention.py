import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout) 

    def forward(self, q, k, v, mask):
        batch_size = q.size(0)
        q = self.W_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attn = self.attn_dropout(F.softmax(scores, dim=-1))
        
        context = torch.matmul(attn, v)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(context)