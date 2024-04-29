import torch
import math
import torch.nn as nn

from gpr.model.module.attention import RotaryEmbedding



class AxialAttention(nn.Module):
    def __init__(self, embed_dim, num_head, use_bias, softmax_scale, dropout,
                 max_position_embeddings, orientation) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        self.dropout = dropout

        assert orientation in ['per_row', 'per_column']
        self.orientation = orientation

        self.num_head = num_head
        assert self.embed_dim % num_head == 0, "self.kdim must be divisible by num_head"
        self.head_dim = self.embed_dim // num_head
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        if softmax_scale:
            self.scale = self.head_dim ** (-0.5)
        else:
            self.scale = None

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=use_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)

        self.rotary_embedding = RotaryEmbedding(
            dim=embed_dim // num_head,
            end=max_position_embeddings,
        )

    def forward(self, tbl_act, tbl_mask):

        assert len(tbl_act.shape) == 4

        if self.orientation == 'per_column':
            tbl_act = torch.swapaxes(tbl_act, -2, -3)
            if tbl_mask is not None:
                tbl_mask = torch.swapaxes(tbl_mask, -1, -2)

        batch_size = tbl_act.shape[0]
        seqlenX = tbl_act.shape[1]
        seqlenY = tbl_act.shape[2]
        extended_batch_size = batch_size * seqlenX


        query, key, value = self.Wqkv(tbl_act).chunk(3, dim=-1)

        query = query.contiguous().view(extended_batch_size, seqlenY, self.num_head, self.head_dim)
        key = key.contiguous().view(extended_batch_size, seqlenY, self.num_head, self.head_dim)
        query = self.rotary_embedding(query, position_ids=None)
        key = self.rotary_embedding(key, position_ids=None)
        query = query.view(extended_batch_size, seqlenY, -1)
        key = key.view(extended_batch_size, seqlenY, -1)

        value = value.contiguous().view(extended_batch_size, seqlenY, -1)

        if self.training:
            dropout_p = self.dropout
        else:
            dropout_p = 0

        attn_out = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=tbl_mask, dropout_p=dropout_p,
                                                                    is_causal=False, scale=self.scale)

        attn_out = attn_out.view(batch_size, seqlenX, seqlenY, self.num_head * self.head_dim)

        tbl_act = self.out_proj(attn_out)

        if self.orientation == 'per_column':
            tbl_act = torch.swapaxes(tbl_act, -2, -3)

        return tbl_act
