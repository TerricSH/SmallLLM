# SmallLLM

一个轻量级的大语言模型实现，基于 PyTorch，参考 LLaMA 架构设计。

## 架构特性

- **RMSNorm** — 均方根归一化，替代传统 LayerNorm
- **RoPE (Rotary Position Embedding)** — 旋转位置编码，支持长序列外推
- **SwiGLU FFN** — 使用 SiLU 门控的前馈网络
- **Causal Attention** — 因果注意力掩码，支持自回归生成
- **Weight Tying** — 词嵌入与输出层权重绑定

## 快速开始

```python
import torch
from SmallLLM import SmallLLM, SmallLLMConfig

config = SmallLLMConfig(
    head_count=8,
    layer_count=16,
    dim=512,
    vocab_size=32000,
)

model = SmallLLM(config)

# 前向传播
input_ids = torch.randint(0, config.vocab_size, (2, 128))  # [batch, seq_len]
logits = model(input_ids)  # [batch, seq_len, vocab_size]

# 带标签计算 loss
targets = torch.randint(0, config.vocab_size, (2, 128))
logits, loss = model(input_ids, targets)
print(f"Loss: {loss.item():.4f}")
```

## 依赖

- Python 3.8+
- PyTorch 2.0+

## 许可

[Apache License 2.0](LICENSE)
