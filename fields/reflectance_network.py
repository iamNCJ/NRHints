from dataclasses import dataclass

import torch
from torch import nn

from fields.encodings import NeRFEncoding


@dataclass(frozen=True)
class ReflectanceNetConfig:
    """Configuration for the reflectance network."""

    d_hidden: int = 256
    """Number of hidden units in the MLP."""
    n_layers: int = 4
    """Number of hidden layers in the MLP."""
    weight_norm: bool = True
    """Whether to use weight normalization."""
    multi_res: int = 4
    """Number of frequencies to use in the input embedding."""
    squeeze_out: bool = True
    """Whether to squeeze the output to [0, 1]."""


class ReflectanceNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 d_in,
                 d_out,
                 config: ReflectanceNetConfig,
                 shadow_hint=True,
                 specular_hint=True,
                 specular_hint_len=6):
        super().__init__()

        self.squeeze_out = config.squeeze_out
        self.shadow_hint = shadow_hint
        self.specular_hint = specular_hint
        dims = [d_in + d_feature] + [config.d_hidden for _ in range(config.n_layers)] + [d_out]

        self.embed_view_pl_fn = NeRFEncoding(
            in_dim=3,
            num_frequencies=config.multi_res,
            include_input=True
        )
        dims[0] += (self.embed_view_pl_fn.get_out_dim() - 3) * 2  # also add PE for point light
        if self.shadow_hint:
            self.embed_vis_fn = NeRFEncoding(in_dim=1, num_frequencies=config.multi_res, include_input=True)
            dims[0] += self.embed_vis_fn.get_out_dim() - 1  # add PE for visibility
        if self.specular_hint:
            self.embed_spec_fn = \
                NeRFEncoding(in_dim=specular_hint_len, num_frequencies=config.multi_res, include_input=True)
            dims[0] += self.embed_spec_fn.get_out_dim() - specular_hint_len  # add PE for specular cue

        self.num_layers = len(dims)

        for layer_idx in range(0, self.num_layers - 1):
            out_dim = dims[layer_idx + 1]
            lin = nn.Linear(dims[layer_idx], out_dim)

            if config.weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(layer_idx), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors, point_lights, visibilities=None, specular_cue=None):
        if self.embed_view_pl_fn is not None:
            view_dirs = self.embed_view_pl_fn(view_dirs)
            point_lights = self.embed_view_pl_fn(point_lights)
        if self.shadow_hint:
            visibilities = self.embed_vis_fn(visibilities)
        if self.specular_hint:
            specular_cue = self.embed_spec_fn(specular_cue)

        rendering_input = [points, view_dirs, normals, point_lights, feature_vectors]

        if self.shadow_hint:
            rendering_input.append(visibilities)
        if self.specular_hint:
            rendering_input.append(specular_cue)

        x = torch.cat(rendering_input, dim=-1)

        for layer_idx in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer_idx))

            x = lin(x)

            if layer_idx < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x
