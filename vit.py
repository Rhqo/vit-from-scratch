import torch
from torch import nn
import numpy as np

class ViTConfig:
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 num_channels=3,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.0,
                 attention_probs_dropout_prob=0.0,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 qkv_bias=True):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias

class ViTPatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.hidden_size = config.hidden_size

        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, pixel_values):
        batch_size, channels, height, width = pixel_values.shape
        # (batch_size, hidden_size, num_patches_h, num_patches_w) -> (batch_size, hidden_size, num_patches)
        # -> (batch_size, num_patches, hidden_size)
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings

class ViTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.patch_embeddings = ViTPatchEmbeddings(config)
        self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        embeddings = self.patch_embeddings(pixel_values)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class ViTSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class ViTSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class ViTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)

    def forward(self, hidden_states, output_attentions=False):
        self_outputs = self.attention(hidden_states, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class ViTIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class ViTOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states

class ViTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, output_attentions=False):
        attention_outputs = self.attention(self.layernorm_before(hidden_states), output_attentions)
        attention_output = attention_outputs[0]

        hidden_states = attention_output + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        layer_output = self.output(layer_output, hidden_states)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs

class ViTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, output_attentions=False, output_hidden_states=False):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, output_attentions)
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = ViTEmbeddings(config)
        self.encoder = ViTEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.classifier = nn.Linear(config.hidden_size, 2) # Example for 2 classes  

    def forward(self, pixel_values, output_attentions=False, output_hidden_states=False):
        embedding_output = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(embedding_output, output_attentions, output_hidden_states)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
    
        # logits = self.classifier(sequence_output[:, 0, :])
        # return (logits,) + encoder_outputs[1:]

        cls_token_output = sequence_output[:, 0, :]
    
        return (cls_token_output,) + encoder_outputs[1:]
    

if __name__ == '__main__':
    config = ViTConfig()
    # Create a dummy input tensor
    dummy_images = torch.randn(1, 3, 224, 224)

    # Test VisionTransformer
    model = VisionTransformer(config)

    # logits = model(dummy_images)[0]
    # print("Shape of logits:", logits.shape)

    cls_token_output = model(dummy_images)[0]
    print("Shape of CLS token output:", cls_token_output.shape)
