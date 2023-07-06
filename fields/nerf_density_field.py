from dataclasses import dataclass, field
from typing import List

import torch

import torch.nn.functional as F
from torch import nn

from fields.encodings import NeRFEncoding


@dataclass(frozen=True)
class NeRFConfig:
    """Configuration for the NeRF density field."""

    d_hidden: int = 256
    """Number of hidden units in the MLP."""
    n_layers: int = 8
    """Number of hidden layers in the MLP."""
    multi_res: int = 10
    """Number of frequencies to use in the input embedding."""
    multi_res_view: int = 4
    """Number of frequencies to use in the input embedding."""
    skips: List[int] = field(default_factory=lambda: [4])
    """Indices of layers to skip the input embedding."""


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
# added pl-naive condition for relighting
class NeRF(nn.Module):
    def __init__(
            self,
            d_in=4,
            d_in_view=6,
            config: NeRFConfig = NeRFConfig(),
    ):
        super(NeRF, self).__init__()
        self.D = config.n_layers
        self.W = config.d_hidden
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 6
        self.embed_fn = None
        self.embed_fn_view = None

        self.embed_fn = NeRFEncoding(in_dim=d_in, num_frequencies=config.multi_res, include_input=True)
        self.input_ch = self.embed_fn.get_out_dim()

        self.embed_fn_view = NeRFEncoding(in_dim=d_in_view, num_frequencies=config.multi_res_view, include_input=True)
        self.input_ch_view = self.embed_fn_view.get_out_dim()

        self.skips = config.skips

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, self.W)] +
            [nn.Linear(self.W, self.W) if i not in self.skips else
             nn.Linear(self.W + self.input_ch, self.W) for i in range(self.D - 1)])

        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + self.W, self.W // 2)])

        self.feature_linear = nn.Linear(self.W, self.W)
        self.alpha_linear = nn.Linear(self.W, 1)
        self.rgb_linear = nn.Linear(self.W // 2, 3)

    def forward(self, input_pts, input_views, input_pls):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_view_pls = torch.cat([input_views, input_pls], dim=-1)
            input_views = self.embed_fn_view(input_view_pls)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], -1)

        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        return alpha, rgb
