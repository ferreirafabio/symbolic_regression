import torch.nn as nn

from gpr.model.module.attention import AttentionLayer
from gpr.model.module.feed_forward import FeedForward
from gpr.model.module.norm import RMSNorm


class DecoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        ff_dim = int(config.ff_factor * config.model_dim)

        self.residual_in_fp32 = True

        self.dropout1 = nn.Dropout(config.resi_dropout)
        self.dropout2 = nn.Dropout(config.resi_dropout)
        self.dropout3 = nn.Dropout(config.resi_dropout)

        if config.rms_norm:
            self.norm1 = RMSNorm(config.model_dim, eps=config.ln_eps, add_unit_offset=config.add_unit_offset)
            self.norm2 = RMSNorm(config.model_dim, eps=config.ln_eps, add_unit_offset=config.add_unit_offset)
            self.norm3 = RMSNorm(config.model_dim, eps=config.ln_eps, add_unit_offset=config.add_unit_offset)
            self.norm4 = RMSNorm(config.model_dim, eps=config.ln_eps, add_unit_offset=config.add_unit_offset)
        else:
            self.norm1 = nn.LayerNorm(config.model_dim, eps=config.ln_eps, bias=config.use_bias,
                                      elementwise_affine=config.learn_ln)
            self.norm2 = nn.LayerNorm(config.model_dim, eps=config.ln_eps, bias=config.use_bias,
                                      elementwise_affine=config.learn_ln)
            self.norm3 = nn.LayerNorm(config.model_dim, eps=config.ln_eps, bias=config.use_bias,
                                      elementwise_affine=config.learn_ln)
            self.norm4 = nn.LayerNorm(config.model_dim, eps=config.ln_eps, bias=config.use_bias,
                                      elementwise_affine=config.learn_ln)

        self.cross_attn = AttentionLayer(config.model_dim, config.num_head, is_causal=True, is_cross=True,
                                         use_bias=config.use_bias, softmax_scale=config.softmax_scale,
                                         dropout=config.attn_dropout,
                                         max_position_embeddings=config.max_len)

        self.self_attn = AttentionLayer(config.model_dim, config.num_head, is_causal=True, is_cross=False,
                                        use_bias=config.use_bias, softmax_scale=config.softmax_scale,
                                        dropout=config.attn_dropout,
                                        max_position_embeddings=config.max_len)

        self.transition = FeedForward(config.model_dim, ff_dim, use_bias=config.use_bias, activation=config.activation,
                                      glu=config.glu)

    def forward(self, self_states, cross_states, attn_mask=None):

        hidden_states = self_states + self.dropout1(
            self.cross_attn(self.norm1(cross_states), self.norm2(self_states), attn_mask=attn_mask))

        hidden_states = hidden_states + self.dropout2(
            self.self_attn(self.norm3(hidden_states), None, attn_mask=attn_mask))

        hidden_states = hidden_states + self.dropout3(self.transition(self.norm4(hidden_states)))

        return hidden_states
