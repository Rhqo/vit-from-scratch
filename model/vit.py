import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass

@dataclass
class ViTConfig:
    # Embedding
    num_channels: int = 3
    embed_dim: int = 256
    image_size: int = 32
    patch_size: int = 4
    # EncoderBlock
    num_attention_heads: int = 8
    attention_dropout: float = 0.0
    # Encoder
    num_encoder_blocks: int = 6
    # MLP
    mlp_hidden_dim: int = 256*2
    mlp_dropout: float = 0.0
    # LayerNorm
    layer_norm_eps: float = 1e-6
    # Training
    batch_size = 32
    epochs = 10
    learning_rate = 3e-4
    num_classes = 10

class PatchEmbedding(nn.Module):
    def __init__(self, config = ViTConfig):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=config.num_channels,
                              out_channels=config.embed_dim,
                              kernel_size=config.patch_size,
                              stride=config.patch_size)
        num_patches = (config.image_size // config.patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + num_patches, config.embed_dim))

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        x = self.proj(x) # (B, E, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) # (B, N, E)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x
    
class MLP(nn.Module):
    def __init__(self, config = ViTConfig):
        super().__init__()
        self.fc1 = nn.Linear(in_features=config.embed_dim,
                             out_features=config.mlp_hidden_dim)
        self.fc2 = nn.Linear(in_features=config.mlp_hidden_dim,
                             out_features=config.embed_dim)
        self.dropout = nn.Dropout(config.mlp_dropout)

    def forward(self, x):
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config = ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = nn.MultiheadAttention(config.embed_dim, config.num_attention_heads, config.attention_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, config = ViTConfig):
        super().__init__()
        self.patch_embed = PatchEmbedding()
        self.encoder = nn.Sequential(*[
            TransformerEncoderLayer()
            for _ in range(config.num_encoder_blocks)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)
