# model.py
import torch
import torch.nn as nn
import math
from typing import Optional

class TransformerBlock(nn.Module):
    def __init__(self, config: 'ModelConfig'):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.d_model, 
            config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model)
        )
        self.layer_norm1 = nn.LayerNorm(config.d_model)
        self.layer_norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attended = self.attention(x, x, x, attn_mask=attention_mask)[0]
        x = self.layer_norm1(x + self.dropout(attended))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        
        return x

class HindiTransformer(nn.Module):
    def __init__(self, config: 'ModelConfig'):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_encoding = self._create_position_encoding()
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        self.output_layer = nn.Linear(config.d_model, config.vocab_size)
        
    def _create_position_encoding(self) -> torch.Tensor:
        position = torch.arange(self.config.max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.config.d_model, 2) * 
                           (-math.log(10000.0) / self.config.d_model))
        pos_encoding = torch.zeros(self.config.max_sequence_length, self.config.d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Embed tokens and add positional encoding
        x = self.embedding(input_ids)
        x = x + self.position_encoding[:, :x.size(1), :].to(x.device)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
            
        # Output projection
        return self.output_layer(x)