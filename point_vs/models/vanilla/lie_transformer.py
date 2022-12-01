import torch
from eqv_transformer.eqv_attention import GlobalPool, EquivariantTransformerBlock
from lie_conv.lieGroups import SE3
from lie_conv.utils import Pass
from torch import nn

from point_vs.global_objects import DEVICE
from point_vs.models.vanilla.pnn_vanilla_base import PNNVanillaBase


class EquivariantTransformer(PNNVanillaBase):
    """Adapted from https://github.com/anonymous-code-0/lie-transformer"""

    def _get_y_true(self, y):
        return y.to(DEVICE)

    def prepare_input(self, x):
        return tuple([ten.to(DEVICE) for ten in x])

    def build_net(self, dim_input, dim_output, dim_hidden, num_layers,
                  num_heads, global_pool_mean=True, group=SE3(0.2),
                  liftsamples=1, block_norm="layer_pre", kernel_norm="none",
                  kernel_type="mlp", kernel_dim=16, kernel_act="swish", nbhd=0,
                  fill=1.0, attention_fn="norm_exp", feature_embed_dim=None,
                  max_sample_norm=None, lie_algebra_nonlinearity=None,
                  **kwargs):

        if isinstance(dim_hidden, int):
            dim_hidden = [dim_hidden] * (num_layers + 1)

        if isinstance(num_heads, int):
            num_heads = [num_heads] * num_layers

        attention_block = lambda dim, n_head: EquivariantTransformerBlock(
            dim, n_head, group, block_norm=block_norm, kernel_norm=kernel_norm,
            kernel_type=kernel_type, kernel_dim=kernel_dim,
            kernel_act=kernel_act, mc_samples=nbhd, fill=fill,
            attention_fn=attention_fn, feature_embed_dim=feature_embed_dim,
        )

        layers = [Pass(nn.Linear(dim_input, dim_hidden[0]), dim=1),
                  *[attention_block(dim_hidden[i], num_heads[i])
                    for i in range(num_layers)]]

        if self.linear_gap:
            layers += [
                Pass(nn.Linear(dim_hidden[-1], dim_output), dim=1),
                GlobalPool(mean=global_pool_mean),
            ]
        else:
            layers += [
                GlobalPool(mean=global_pool_mean),
                nn.Linear(dim_hidden[-1], dim_output)
            ]

        layers = nn.Sequential(*layers)

        self.group = group
        self.liftsamples = liftsamples
        self.max_sample_norm = max_sample_norm

        self.lie_algebra_nonlinearity = lie_algebra_nonlinearity
        if lie_algebra_nonlinearity is not None:
            if lie_algebra_nonlinearity == 'tanh':
                self.lie_algebra_nonlinearity = nn.Tanh()
            else:
                raise ValueError('{} is not a supported nonlinearity'.format(
                    lie_algebra_nonlinearity))

        return layers

    def forward(self, x):
        if self.max_sample_norm is None:
            lifted_data = self.group.lift(x, self.liftsamples)
        else:
            lifted_data = [
                torch.tensor(self.max_sample_norm * 2, device=x[0].device),
                0,
                0,
            ]
            while lifted_data[0].norm(dim=-1).max() > self.max_sample_norm:
                lifted_data = self.group.lift(x, self.liftsamples)

        if self.lie_algebra_nonlinearity is not None:
            lifted_data = list(lifted_data)
            pairs_norm = lifted_data[0].norm(dim=-1) + 1e-6
            lifted_data[0] = lifted_data[0] * (
                    self.lie_algebra_nonlinearity(pairs_norm / 7) / pairs_norm
            ).unsqueeze(-1)

        return self.layers(lifted_data)
