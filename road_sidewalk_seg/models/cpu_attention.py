from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as functional
from transformers.models.swin.modeling_swin import SwinSelfAttention


def _sdpa_forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
    if self.training or head_mask is not None or output_attentions:
        return self._cpu_eager_forward(hidden_states, attention_mask, head_mask, output_attentions)
    batch_size, token_count, _ = hidden_states.shape
    hidden_shape = (batch_size, token_count, -1, self.attention_head_size)
    query = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
    key = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
    value = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
    bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
    bias = bias.view(token_count, token_count, -1).permute(2, 0, 1).unsqueeze(0)
    if attention_mask is not None:
        window_count = attention_mask.shape[0]
        bias = bias.unsqueeze(0) + attention_mask.unsqueeze(0).unsqueeze(2)
        bias = bias.expand(batch_size // window_count, -1, -1, -1, -1)
        bias = bias.reshape(batch_size, self.num_attention_heads, token_count, token_count)
    context = functional.scaled_dot_product_attention(
        query, key, value, attn_mask=bias, dropout_p=0.0
    )
    context = context.transpose(1, 2).contiguous().view(batch_size, token_count, self.all_head_size)
    return (context,)


def configure_cpu_attention(model: torch.nn.Module, mode: str) -> int:
    if mode not in {"eager", "sdpa"}:
        raise ValueError("CPU attention mode must be eager or sdpa")
    if mode == "sdpa" and not hasattr(functional, "scaled_dot_product_attention"):
        raise RuntimeError("CPU SDPA requires PyTorch >= 2.0; use cpu_attention=eager")
    count = 0
    for module in model.modules():
        if not isinstance(module, SwinSelfAttention):
            continue
        if not hasattr(module, "_cpu_eager_forward"):
            module._cpu_eager_forward = module.forward
        module.forward = (
            MethodType(_sdpa_forward, module) if mode == "sdpa" else module._cpu_eager_forward
        )
        count += 1
    return count
