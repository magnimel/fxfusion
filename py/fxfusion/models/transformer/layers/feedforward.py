import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion_factor: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
    
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))