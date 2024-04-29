from typing import Dict, Optional, Union
import torch
import torch.nn as nn



class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, end: int, theta: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.end = end
        self.theta = theta
        self.freqs_cis: torch.Tensor
        self._initialized_buffer = False

    def init_rotary_embeddings(self, device):
        if self._initialized_buffer is True:
            # Buffer if already initialized
            return
        self.register_buffer(
            "freqs_cis",
            torch.empty(self.end, self.dim // 2, 2, dtype=torch.float, device=device),
            persistent=False,
        )
        if self.freqs_cis.dtype != torch.float:
            self.freqs_cis = self.freqs_cis.to(torch.float)
        assert self.freqs_cis.dtype == torch.float
        freqs = 1.0 / (
                self.theta
                ** (torch.arange(0, self.dim, 2, dtype=torch.float, device=device)[: (self.dim // 2)] / self.dim)
        )
        t = torch.arange(self.end, device=device)
        freqs = torch.outer(t, freqs).float()
        complex_freqs = torch.polar(torch.ones_like(freqs), freqs)
        freqs = torch.view_as_real(complex_freqs)
        self.freqs_cis.copy_(freqs)
        self._initialized_buffer = True

    def forward(
            self,
            x: torch.Tensor,  # [batch_size, seq_length, num_heads, d_qk]
            position_ids: Optional[torch.LongTensor],  # [batch_size, seq_length]
    ):
        batch_size, seq_length, num_heads, inner_dim = x.shape
        while (
                position_ids is not None and position_ids[-1, -1] >= self.end
        ) or seq_length >= self.end:
            self.end *= 2
            self._initialized_buffer = False
        if self._initialized_buffer is False:
            self.init_rotary_embeddings(device=x.device)
        dtype = x.dtype
        assert inner_dim % 2 == 0
        x = x.view(
            batch_size, seq_length, num_heads, inner_dim // 2, 2
        )  # [batch_size, q_length, num_heads, inner_dim]
        if x.dtype == torch.bfloat16:
            x = x.float()
        complex_x = torch.view_as_complex(x)  # [batch_size, q_length, num_heads, inner_dim // 2]
        if position_ids is None:
            freqs_cis = self.freqs_cis[None, :seq_length, None, :]
        else:
            if position_ids[-1, -1] < 0 or position_ids[-1, -1] >= self.end:  # Quick test hopefully
                raise ValueError(f"Position ids must be in the range [0, {self.end}), but got {position_ids}")
            freqs_cis = self.freqs_cis[position_ids][:, :, None, :]
        complex_freqs = torch.view_as_complex(freqs_cis)
        x_out = torch.view_as_real(complex_x * complex_freqs).view(batch_size, seq_length, num_heads, inner_dim)
        return x_out.type(dtype)


class AttentionLayer(nn.Module):

    def __init__(self, embed_dim, num_head, is_causal, is_cross, use_bias, softmax_scale, dropout,
                 max_position_embeddings) -> None:

        super().__init__()
        self.embed_dim = embed_dim
        self.is_causal = is_causal
        self.is_cross = is_cross

        self.dropout = dropout

        self.num_head = num_head
        assert self.embed_dim % num_head == 0, "self.kdim must be divisible by num_head"
        self.head_dim = self.embed_dim // num_head
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        if softmax_scale:
            self.scale = self.head_dim ** (-0.5)
        else:
            self.scale = None

        if is_cross:
            self.Wkv = nn.Linear(embed_dim, 2 * embed_dim, bias=use_bias)
            self.Wq = nn.Linear(embed_dim, 1 * embed_dim, bias=use_bias)
        else:
            self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=use_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)

        self.rotary_embedding = RotaryEmbedding(
            dim=embed_dim // num_head,
            end=max_position_embeddings,
        )

    def forward(self, keyvalue, query, attn_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """

        if self.is_cross:
            key, value = self.Wkv(keyvalue).chunk(2, dim=-1)
            query = self.Wq(query)
        else:
            query, key, value = self.Wqkv(keyvalue).chunk(3, dim=-1)

        query = query.contiguous().view(query.size(0), query.size(1), self.num_head, self.head_dim)
        key = key.contiguous().view(key.size(0), key.size(1), self.num_head, self.head_dim)
        query = self.rotary_embedding(query, position_ids=None)
        key = self.rotary_embedding(key, position_ids=None)
        query = query.view(query.size(0), query.size(1), -1)
        key = key.view(key.size(0), key.size(1), -1)

        if self.training:
            dropout_p = self.dropout
        else:
            dropout_p = 0

        attn_out = torch.nn.functional.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p,
                                                                    is_causal=self.is_causal, scale=self.scale)

        return self.out_proj(attn_out)
