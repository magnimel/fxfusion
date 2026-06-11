import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        assert encoder is not None and decoder is not None
        
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encoder(src, src_mask)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
            
