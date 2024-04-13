import math

import torch
import torch.nn as nn

from gpr.model.module.embedding import PosEmbedding
from gpr.model.module.decoder_block import DecoderBlock
from gpr.model.module.encoder_block import EncoderBlock
from gpr.model.module.norm import RMSNorm

class GPRTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.trg_embed = PosEmbedding(config.trg_vocab_size, config.model_dim, config.max_len, config.pos_embedding,
                                      config.rel_pos_enc, config.initializer_range)
        self.trg_output = self.trg_embed.embed_seq

        self.embed_significand = nn.Linear(1, config.model_dim)
        self.embed_exponent = nn.Linear(1, config.model_dim)

        self.encoder = nn.ModuleList()
        for _ in range(config.n_layers):
            self.encoder.append(EncoderBlock(config))

        self.decoder = nn.ModuleList()
        for _ in range(config.n_layers):
            self.decoder.append(DecoderBlock(config))


        if config.rms_norm:
            self.trg_norm = RMSNorm(config.model_dim, eps=config.ln_eps, add_unit_offset=config.add_unit_offset)
        else:
            self.trg_norm = nn.LayerNorm(config.model_dim, eps=config.ln_eps, bias=config.use_bias, elementwise_affine=config.learn_ln)


        self.initialize(config.initializer_range, config.trg_vocab_size, config.last_zero)

    def initialize(self, initializer_range, vocab_size, last_zero):

        for n, p in self.named_parameters():
            if 'bias' in n:
                nn.init.zeros_(p)
            elif 'norm' in n or 'ln' in n:
                continue
            elif p.shape == torch.Size([1]):
                continue
            elif 'out' in n and last_zero:
                nn.init.zeros_(p)
            elif 'embed_seq' in n:
                if initializer_range:
                    nn.init.normal_(p, mean=0.0, std=initializer_range)
                else:
                    nn.init.normal_(p, mean=0.0, std=1.0/math.sqrt(vocab_size))
            else:
                if initializer_range:
                    nn.init.normal_(p, mean=0.0, std=initializer_range)
                else:
                    nn.init.xavier_uniform_(p)




    def make_decoder_mask(self, trg_embed, trg_len):
        mask = torch.arange(trg_embed.size()[1], device=trg_embed.device).expand(trg_embed.shape[:2]) < trg_len.unsqueeze(1)
        mask = mask.unsqueeze(-1)

        causal_mask = torch.triu(torch.ones((1, trg_embed.size()[1], trg_embed.size()[1]), dtype=torch.bool, device=trg_embed.device), diagonal=1)
        causal_mask = causal_mask == 0
        decoder_mask = mask & causal_mask

        assert isinstance(decoder_mask, torch.BoolTensor) or isinstance(decoder_mask, torch.cuda.BoolTensor)

        return torch.bitwise_not(decoder_mask)


    def forward(self, tbl_significand, tbl_exponent, feature_len, sample_len, trg_shifted_equation, trg_len):


        tbl_latent = self.embed_significand(tbl_significand) + self.embed_exponent(tbl_exponent)

        tbl_mask = None # TODO: Implement mask

        for i in range(len(self.encoder)):
            layer = self.encoder[i]
            tbl_latent = layer(tbl_states=tbl_latent, tbl_mask=tbl_mask)

        encoder_latent = tbl_latent[:,:,0,:]

        trg_latent = self.trg_embed(trg_shifted_equation)


        for i in range(len(self.decoder)):
            layer = self.decoder[i]
            trg_latent = layer(self_states=trg_latent, cross_states=encoder_latent)

        logits = self.trg_output(self.trg_norm(trg_latent))

        return logits


if __name__ == '__main__':

    from collections import namedtuple

    config = {
        "trg_vocab_size": 200,
        "model_dim": 128,
        "max_len": 1000,
        "num_head": 2,
        "ff_factor": 4,
        "activation": "relu",
        "glu": True,
        "softmax_scale": True,
        "pos_embedding": True,
        "rel_pos_enc": False,
        "initializer_range": 0.02,
        "seq_vocab_size": 1000,
        "n_layers": 6,
        "rms_norm": False,
        "resi_dropout": 0.1,
        "attn_dropout": 0.1,
        "ln_eps": 1e-5,
        "add_unit_offset": False,
        "use_bias": True,
        "learn_ln": True,
        "last_zero": False
    }

    config = namedtuple("Config", config.keys())(*config.values())
    model = GPRTransformer(config).to('cuda')
    tbl_significand = torch.randn((32, 512, 5, 1)).to('cuda')
    tbl_exponent = torch.randint(-3, 10, (32, 512, 5, 1)).to(torch.float).to('cuda')

    trg_equation = torch.randint(0, 200, (32, 100)).to('cuda')
    feature_len = torch.ones(32, dtype=torch.long).to('cuda') * 5
    sample_len = torch.ones(32, dtype=torch.long).to('cuda') * 512
    trg_len = torch.ones(32, dtype=torch.long).to('cuda') * 100
    logits = model(tbl_significand, tbl_exponent, feature_len, sample_len, trg_equation, trg_len)
    print(logits.shape)
    print(logits)
