"""
Transformer for the six prediction tasks that the MLP runs (input genes -> target genes).

Only the network differs from the MLP: the data preparation, cell split, gene lists,
transformations, MSE loss, train/validation epoch selection, train+validation refit,
test evaluation and every output file come from mlp_common.py unchanged. use_transformer()
swaps mlp_common's model hook so those shared functions build this network instead.

The network follows Step2_train_Transformer_denoise.py: the input vector is split into
n_tokens chunks, each chunk is projected to d_model, a learnable positional embedding is
added, a GELU Transformer encoder is applied, and the flattened tokens are projected to
output_dim with a final ReLU (non-negative output, same as the MLP).
"""

import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlp_common


class TransformerPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, n_tokens=32, d_model=64, nhead=4,
                 num_layers=1, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_dim, self.n_tokens = input_dim, n_tokens
        self.chunk_size = math.ceil(input_dim / n_tokens)
        self.padded_input_dim = self.chunk_size * n_tokens
        self.input_proj = nn.Linear(self.chunk_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=dim_feedforward, dropout=dropout,
                                           batch_first=True, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(n_tokens * d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim),
            nn.ReLU(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        b = x.size(0)
        if self.input_dim < self.padded_input_dim:
            x = torch.cat([x, x.new_zeros(b, self.padded_input_dim - self.input_dim)], dim=1)
        x = self.input_proj(x.view(b, self.n_tokens, self.chunk_size)) + self.pos_embedding
        x = self.transformer_encoder(x)
        return self.output_proj(x.reshape(b, -1))


def _build(cfg, input_dim, output_dim, dropout):
    return TransformerPredictor(input_dim, output_dim, n_tokens=cfg["n_tokens"],
                                d_model=cfg["d_model"], nhead=cfg["nhead"],
                                num_layers=cfg["num_layers"],
                                dim_feedforward=cfg["dim_feedforward"], dropout=dropout)


def add_transformer_args(p):
    # defaults match the Transformer_denoise settings in run_pbmc_general_mse_reverse.sh
    p.add_argument("--task", required=True, choices=["within", "within_raw", "cross"])
    p.add_argument("--n_tokens", type=int, default=32)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dim_feedforward", type=int, default=128)
    return p


def use_transformer(args=None):
    """Point mlp_common's model hook at the Transformer. Training passes its args so the
    architecture is recorded in final_config.json; prediction reads it back from there."""
    mlp_common.ARCH_NAME = "Transformer"
    mlp_common.build_model = _build
    if args is not None:
        mlp_common.MODEL_SPEC = dict(n_tokens=args.n_tokens, d_model=args.d_model,
                                     nhead=args.nhead, num_layers=args.num_layers,
                                     dim_feedforward=args.dim_feedforward)