# Small LLM: A lightweight language model. 

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class SmallLLMConfig:
    def __init__(
        self,
        head_count: int = 8,
        layer_count: int = 16,
        dim: int = 512,
        hidden_dim: int = 2048,
        max_seq_len: int = 2048,
        attention_dropout: float = 0.0,
        norm_eps: float = 1e-6,
        vocab_size: int = 32000,
    ):
        self.head_count = head_count
        self.layer_count = layer_count
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.attention_dropout = attention_dropout
        self.norm_eps = norm_eps
        self.vocab_size = vocab_size

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps # 防止除以零
        self.weight = nn.Parameter(torch.ones(dim)) # 可训练矩阵，增加差别的表达（权重）
    
    def _norml(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) # 求平方根均数 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        retVal = self._norml(x.float()).type_as(x) 
        return retVal * self.weight 

def precompute_rope_frequencies(dim: int, seq_length: int, theta: float = 10000, device: Optional[torch.device] = None):
    # 角速度
    omiga = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    # 位置 矩阵
    t = torch.arange(seq_length, device=omiga.device, dtype=torch.float32)
    # 获得角度频率 
    freqs = torch.outer(t, omiga)
    # 返回freqs
    return torch.polar(torch.ones_like(freqs), freqs)

def reshape_for_broadcast(freqs_cis:torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    # 改一下形状， 把freqs的结构里加两列，匹配xq形状
    ndim = xq.ndim
    # 检查一下维度内容是否对齐 批次，序列长度，头数，头维度
    if (freqs_cis.shape != (xq.shape[1], xq.shape[-1])):
        raise ValueError(f"freqs_cis shape {freqs_cis.shape} does not match expected shape {(xq.shape[1], xq.shape[-1])}")

    shape = [d if i== 1 or i == ndim -1 else 1 for i, d in enumerate(xq.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(wq: torch.Tensor, wk: torch.Tensor, freqs_cis:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # 实数转虚数
    xq_i = torch.view_as_complex(wq.float().reshape(*wq.shape[:-1], -1, 2))
    xk_i = torch.view_as_complex(wk.float().reshape(*wk.shape[:-1], -1, 2))
    # 旋转
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_i)
    # 虚数转实数
    wq_out = torch.view_as_real(xq_i * freqs_cis).flatten(3)
    wk_out = torch.view_as_real(xk_i * freqs_cis).flatten(3)
    return wq_out.type_as(wq), wk_out.type_as(wk)

class SmallLLM(nn.Module):
    def __init__(self, config: SmallLLMConfig) :
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            LayerBlock(config) for _ in range(config.layer_count)
        ])
        self.final_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight  # 权重绑定

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        if x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            hidden_states = self.tok_embeddings(x.long())
        elif x.is_floating_point() and x.dim() == 3 and x.size(-1) == self.config.dim:
            hidden_states = x
        else:
            raise ValueError("Expected token ids [B, L] or hidden states [B, L, dim]")

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )
        return logits, loss
    
class LayerBlock(nn.Module):
    def __init__(self, config: SmallLLMConfig):
        super().__init__()
        self.Attention = MHA(config)
        self.FFN = FFN(config.dim, config.hidden_dim, None)
        self.attention_norm = RMSNorm(config.dim,eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim,eps=config.norm_eps)
    def forward(self, x) -> torch.Tensor:
        hidden_states = x + self.Attention(self.attention_norm(x))
        retVal = hidden_states + self.FFN(self.ffn_norm(hidden_states))
        return retVal

# Attention
class MHA(nn.Module):
    def __init__(self, config: SmallLLMConfig):
        super().__init__()
        self.config = config
        if config.dim % config.head_count != 0:
            raise ValueError(f"dim ({config.dim}) must be divisible by head_count ({config.head_count})")
        self.head_count = config.head_count
        self.head_dim = config.dim // config.head_count
        self.wq = nn.Linear(config.dim, config.dim, bias=False)
        self.wk = nn.Linear(config.dim, config.dim, bias=False)
        self.wv = nn.Linear(config.dim, config.dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.attention_dropout = config.attention_dropout
        self.freqs_cis: torch.Tensor
        self.register_buffer(
            "freqs_cis",
            precompute_rope_frequencies(self.head_dim, config.max_seq_len),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = x.shape
        freqs_cis = self.freqs_cis[:seq_length]
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(batch_size, seq_length, self.head_count , self.head_dim)
        xk = xk.view(batch_size, seq_length, self.head_count , self.head_dim)
        xv = xv.view(batch_size, seq_length, self.head_count , self.head_dim)
        #RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        xq = xq.transpose(1, 2) # (batch_size, head_count, seq_length, head_dim)
        xk = xk.transpose(1, 2) 
        xv = xv.transpose(1, 2) 
        # 计算注意力分数
        scores = torch.matmul(xq, xk.transpose(2,3)) / math.sqrt(self.head_dim)

        causal_mask = torch.triu(
            torch.full((seq_length, seq_length), float("-inf"), device=x.device, dtype=scores.dtype),
            diagonal=1,
        )

        scores = scores + causal_mask 
        scores = F.softmax(scores.float(), dim=-1).type_as(x)

        if self.attention_dropout > 0.0 and self.training:
            scores = F.dropout(scores, p=self.attention_dropout)
        
        x = torch.matmul(scores, xv)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.head_count * self.head_dim)
        return self.wo(x)

# feed forward 升维 降维
class FFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiplier: Optional[float]):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = 256 * ((hidden_dim + 255) // 256)  # 对齐到 256 的倍数，提升 GPU 效率
        if multiplier is not None:
            hidden_dim = int(multiplier * hidden_dim)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up =  nn.Linear(dim, hidden_dim, bias=False)
        self.w_down =  nn.Linear(hidden_dim, dim, bias=False)
    def forward(self, x ):
        # SwiGLU 
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)).type_as(x)
