"""Shared multimodal model architecture."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from SmallLLM import SmallLLM


class VisionMHA(nn.Module):
    """Multi-head attention with an optional attention-score penalty."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.embed_dim = dim
        self.head_count = heads
        self.head_dim = dim // heads
        self.in_proj_weight = nn.Parameter(torch.randn(3 * dim, dim) * 0.02)
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * dim))
        self.out_proj = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        penalty_threshold: Optional[float] = None,
    ):
        batch_size, sequence_length, dim = x.shape
        q, k, v = F.linear(
            x,
            self.in_proj_weight,
            self.in_proj_bias,
        ).chunk(3, dim=-1)
        q = q.view(
            batch_size,
            sequence_length,
            self.head_count,
            self.head_dim,
        ).transpose(1, 2)
        k = k.view(
            batch_size,
            sequence_length,
            self.head_count,
            self.head_dim,
        ).transpose(1, 2)
        v = v.view(
            batch_size,
            sequence_length,
            self.head_count,
            self.head_dim,
        ).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q * scale, k.transpose(2, 3))

        penalty = scores.new_zeros((), dtype=torch.float32)
        if penalty_threshold is not None and penalty_threshold > 0.0:
            overflow = F.relu(scores.float().abs() - penalty_threshold)
            active = (overflow > 0).sum().clamp_min(1)
            penalty = (
                overflow.square().sum()
                / active
                / (penalty_threshold * penalty_threshold)
            )

        attention = F.softmax(scores, dim=-1).type_as(v)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            dim,
        )
        return self.out_proj(output), penalty


class VisionEncoderLayer(nn.Module):
    """Pre-normalized transformer encoder layer."""

    def __init__(self, dim: int, heads: int, hidden_dim: int):
        super().__init__()
        self.self_attn = VisionMHA(dim, heads)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(0.0)
        self.dropout2 = nn.Dropout(0.0)

    def forward_with_penalty(self, x: torch.Tensor, threshold: float):
        residual = x
        attention, penalty = self.self_attn(self.norm1(x), threshold)
        x = residual + self.dropout1(attention)
        x = x + self.dropout2(
            self.linear2(F.gelu(self.linear1(self.norm2(x))))
        )
        return x, penalty

    def forward(self, x: torch.Tensor):
        residual = x
        attention, _ = self.self_attn(self.norm1(x), None)
        x = residual + self.dropout1(attention)
        return x + self.dropout2(
            self.linear2(F.gelu(self.linear1(self.norm2(x))))
        )


class EncoderBlock(nn.Module):
    """Stacked vision encoder layers followed by a final normalization."""

    def __init__(self, layer_list, final_norm):
        super().__init__()
        self.layers = layer_list
        self.norm = final_norm

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class VisionEncoder(nn.Module):
    def __init__(
        self,
        llm_dim,
        image_size=224,
        patch_size=16,
        vision_dim=512,
        layers=4,
        heads=8,
        patch_activation_penalty_weight=0.0,
        patch_activation_penalty_threshold=0.0,
    ):
        super().__init__()
        token_count = (image_size // patch_size) ** 2
        if image_size % patch_size or vision_dim % heads:
            raise ValueError("invalid vision dimensions")

        self.patch = nn.Conv2d(3, vision_dim, patch_size, patch_size)
        self.pos = nn.Parameter(torch.randn(1, token_count, vision_dim) * 0.02)
        layer_list = nn.ModuleList([
            VisionEncoderLayer(vision_dim, heads, vision_dim * 4)
            for _ in range(layers)
        ])
        self.blocks = EncoderBlock(layer_list, nn.LayerNorm(vision_dim))
        self.project = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
        self.bounds = nn.Parameter(torch.randn(1, 2, llm_dim) * 0.02)
        self.patch_penalty_weight = patch_activation_penalty_weight
        self.patch_penalty_threshold = patch_activation_penalty_threshold

    @property
    def token_count(self):
        return self.pos.size(1) + 2

    def forward(self, inputs):
        raw = self.patch(inputs)
        patches = raw.flatten(2).transpose(1, 2)
        encoded = self.blocks(patches + self.pos)
        encoded = self.project(encoded)
        bounds = self.bounds.expand(encoded.size(0), -1, -1)
        output = torch.cat((bounds[:, :1], encoded, bounds[:, 1:]), dim=1)

        patch_penalty = raw.new_zeros((), dtype=torch.float32)
        if (
            self.training
            and self.patch_penalty_weight > 0.0
            and self.patch_penalty_threshold > 0.0
        ):
            overflow = F.relu(
                raw.float().abs() - self.patch_penalty_threshold
            )
            active = (overflow > 0).sum().clamp_min(1)
            patch_penalty = (
                overflow.square().sum()
                / active
                / (
                    self.patch_penalty_threshold
                    * self.patch_penalty_threshold
                )
            )
        return output, patch_penalty


class MultimodalSmallLLM(nn.Module):
    def __init__(self, config, vision, patch_activation_penalty_weight=0.0):
        super().__init__()
        self.language_model = SmallLLM(config)
        self.vision_encoder = VisionEncoder(config.dim, **vision)
        self.patch_penalty_weight = patch_activation_penalty_weight

    def forward(self, images, ids, labels=None, vision_fp32=False):
        if vision_fp32:
            with torch.autocast(
                device_type=images.device.type,
                enabled=False,
            ):
                visual, patch_penalty = self.vision_encoder(images)
        else:
            visual, patch_penalty = self.vision_encoder(images)

        text = self.language_model.tok_embeddings(ids)
        visual = visual.to(text.dtype)
        hidden = torch.cat((visual, text), dim=1)

        if hidden.size(1) > self.language_model.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        if labels is None:
            return self.language_model(hidden)

        ignored = labels.new_full((labels.size(0), visual.size(1)), -100)
        targets = torch.cat((ignored, labels), dim=1)
        logits, loss = self.language_model(hidden, targets)
        if self.patch_penalty_weight > 0.0 and patch_penalty.item() > 0:
            loss = loss + self.patch_penalty_weight * patch_penalty
        return logits, loss
