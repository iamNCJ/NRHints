from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from torch import nn

from fields.encodings import NeRFEncoding


@dataclass(frozen=True)
class SDFNetConfig:
    """Configuration for the SDF network."""

    d_in: int = 3
    """Dimensionality of the input."""
    d_out_feat: int = 256
    """Dimensionality of the output."""
    d_hidden: int = 256
    """Number of hidden units in the MLP."""
    n_layers: int = 8
    """Number of hidden layers in the MLP."""
    skip_in: List[int] = field(default_factory=lambda: [4])
    """Indices of layers to skip the input embedding."""
    multi_res: int = 6
    """Number of frequencies to use in the input embedding."""
    init_bias: float = 0.5
    """Initial bias of the output layer, equals the radius of the init sphere."""
    scale: float = 3.0
    """Scale value for the input values."""
    geometric_init: bool = True
    """Whether to use geometric initialization from SAL."""
    weight_norm: bool = True
    """Whether to use weight normalization."""
    inside_outside: bool = False
    """Whether to use inverted SDF values."""


class SDFNetwork(nn.Module):
    def __init__(self, config: SDFNetConfig):
        super(SDFNetwork, self).__init__()

        dims = [config.d_in] + [config.d_hidden for _ in range(config.n_layers)] + [config.d_out_feat + 1]

        self.embed_fn_fine = None

        self.embed_fn_fine = NeRFEncoding(
            in_dim=config.d_in,
            num_frequencies=config.multi_res,
            include_input=True
        )
        dims[0] = self.embed_fn_fine.get_out_dim()

        self.num_layers = len(dims)
        self.skip_in = config.skip_in
        self.scale = config.scale

        bias = config.init_bias * config.scale
        # init input & middle layers
        for layer_idx in range(0, self.num_layers - 2):
            if layer_idx + 1 in self.skip_in:
                out_dim = dims[layer_idx + 1] - dims[0]
            else:
                out_dim = dims[layer_idx + 1]

            lin = nn.Linear(dims[layer_idx], out_dim)

            if config.geometric_init:
                if config.multi_res > 0 and layer_idx == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif config.multi_res > 0 and layer_idx in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if config.weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(layer_idx), lin)

        # init output layers
        output_layers_conf = {
            'sdf': 1,
            'feat': dims[-1] - 1
        }
        for layer_name, out_dim in output_layers_conf.items():
            lin = nn.Linear(dims[-2], out_dim)
            if config.geometric_init:
                if not config.inside_outside:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[-1]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                else:
                    torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[-1]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, bias)
            if config.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "out_" + layer_name, lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        inputs = self.embed_fn_fine(inputs)

        x = inputs
        for layer_idx in range(0, self.num_layers - 2):
            lin = getattr(self, "lin" + str(layer_idx))

            if layer_idx in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if layer_idx < self.num_layers - 2:
                x = self.activation(x)
        sdf = self.out_sdf(x) / self.scale
        feat = self.out_feat(x)
        return torch.cat([sdf, feat], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)[:, 1:]

    def freeze_geometry(self):
        for param_name, param in self.named_parameters():
            if 'lin' in param_name or 'out_sdf' in param_name:
                param.requires_grad = False

    def gradient(self, x):
        x.requires_grad_(True)
        with torch.inference_mode(False), torch.enable_grad():
            y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)
