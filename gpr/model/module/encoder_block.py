import math
import torch
import torch.nn as nn

from gpr.model.module.axial_attention import AxialAttention
from gpr.model.module.axial_dropout import AxialDropout
from gpr.model.module.feed_forward import FeedForward
from gpr.model.module.norm import RMSNorm

class EncoderBlock(nn.Module):


    def __init__(self, config):
        super().__init__()

        ff_dim = int(config.ff_factor * config.model_dim)

        self.residual_in_fp32 = True

        self.dropout1 = AxialDropout(config.resi_dropout, orientation='per_row')
        self.dropout2 = AxialDropout(config.resi_dropout, orientation='per_column')
        self.dropout3 = nn.Dropout(config.resi_dropout)

        if config.rms_norm:
            self.norm1 = RMSNorm(config.model_dim, eps=config.ln_eps, add_unit_offset=config.add_unit_offset)
            self.norm2 = RMSNorm(config.model_dim, eps=config.ln_eps, add_unit_offset=config.add_unit_offset)
            self.norm3 = RMSNorm(config.model_dim, eps=config.ln_eps, add_unit_offset=config.add_unit_offset)
        else:
            self.norm1 = nn.LayerNorm(config.model_dim, eps=config.ln_eps, bias=config.use_bias, elementwise_affine=config.learn_ln)
            self.norm2 = nn.LayerNorm(config.model_dim, eps=config.ln_eps, bias=config.use_bias, elementwise_affine=config.learn_ln)
            self.norm3 = nn.LayerNorm(config.model_dim, eps=config.ln_eps, bias=config.use_bias, elementwise_affine=config.learn_ln)

        self.sample_attn = AxialAttention(config.model_dim, config.num_head, use_bias=config.use_bias, softmax_scale=config.softmax_scale,
                                        dropout=config.attn_dropout,
                                        max_position_embeddings=config.max_len, orientation='per_row')

        self.feature_attn = AxialAttention(config.model_dim, config.num_head,  use_bias=config.use_bias, softmax_scale=config.softmax_scale, dropout=config.attn_dropout,
                                        max_position_embeddings=config.max_len, orientation='per_column')

        self.transition = FeedForward(config.model_dim, ff_dim, use_bias=config.use_bias, activation=config.activation, glu=config.glu)


    def forward(self, tbl_states, tbl_mask=None):

        hidden_states = tbl_states + self.dropout1(self.sample_attn(self.norm1(tbl_states), tbl_mask=tbl_mask))

        hidden_states = hidden_states + self.dropout2(self.feature_attn(self.norm2(hidden_states), tbl_mask=tbl_mask))

        hidden_states = hidden_states + self.dropout3(self.transition(self.norm3(hidden_states)))

        return hidden_states
