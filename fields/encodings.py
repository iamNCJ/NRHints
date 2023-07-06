# Modified from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/field_components/encodings.py

# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Encoding functions
"""

from abc import abstractmethod
from typing import Optional
import logging

import numpy as np
import torch
from torch import nn
from jaxtyping import Float
from typing_extensions import Literal

try:
    # noinspection PyUnresolvedReferences
    import tinycudann as tcnn
    TCNN_EXISTS = True
except ImportError:
    TCNN_EXISTS = False

Tensor = torch.Tensor

logger = logging.getLogger(__name__)


class Encoding(nn.Module):
    """Encode an input tensor. Intended to be subclassed

    Args:
        in_dim: Input dimension of tensor
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        self.in_dim = in_dim

    @abstractmethod
    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """Call forward and returns and processed tensor

        Args:
            in_tensor: the input tensor to process
        """
        raise NotImplementedError

    @abstractmethod
    def get_out_dim(self) -> int:
        """get output dimension of encoding
        """
        raise NotImplementedError


class Identity(Encoding):
    """Identity encoding (Does not modify input)"""

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        return self.in_dim

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        return in_tensor


class ScalingAndOffset(Encoding):
    """Simple scaling and offset to input

    Args:
        in_dim: Input dimension of tensor
        scaling: Scaling applied to tensor.
        offset: Offset applied to tensor.
    """

    def __init__(self, in_dim: int, scaling: float = 1.0, offset: float = 0.0) -> None:
        super().__init__(in_dim)

        self.scaling = scaling
        self.offset = offset

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        return self.in_dim

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        return self.scaling * in_tensor + self.offset


def expected_sin(x_means: Tensor, x_vars: Tensor) -> Tensor:
    """Computes the expected value of sin(y) where y ~ N(x_means, x_vars)

    Args:
        x_means: Mean values.
        x_vars: Variance of values.

    Returns:
        Tensor: The expected value of sin.
    """

    return torch.exp(-0.5 * x_vars) * torch.sin(x_means)


class NeRFEncoding(Encoding):
    """Multi-scale sinusoidal encodings.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
            self, in_dim: int, num_frequencies: int, min_freq_exp: float = 0., max_freq_exp: float = None,
            include_input: bool = False
    ) -> None:
        super().__init__(in_dim)
        if max_freq_exp is None:
            max_freq_exp = num_frequencies - 1

        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_input = include_input

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def forward(
            self,
            in_tensor: Float[Tensor, "*bs input_dim"]
    ) -> Float[Tensor, "*bs output_dim"]:
        """Calculates NeRF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
        Returns:
            Output values will be between -1 and 1
        """
        # in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies, device=in_tensor.device)
        scaled_inputs = in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]
        encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))

        if self.include_input:
            encoded_inputs = torch.cat([in_tensor, encoded_inputs], dim=-1)

        return encoded_inputs


class RFFEncoding(Encoding):
    """Random Fourier Feature encoding. Supports integrated encodings.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoding frequencies
        scale: Std of Gaussian to sample frequencies. Must be greater than zero
        include_input: Append the input coordinate to the encoding
    """

    def __init__(self, in_dim: int, num_frequencies: int, scale: float, include_input: bool = False) -> None:
        super().__init__(in_dim)

        self.num_frequencies = num_frequencies
        if not scale > 0:
            raise ValueError("RFF encoding scale should be greater than zero")
        self.scale = scale
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        b_matrix = torch.normal(mean=0, std=self.scale, size=(self.in_dim, self.num_frequencies))
        self.register_buffer(name="b_matrix", tensor=b_matrix)
        self.include_input = include_input

    def get_out_dim(self) -> int:
        return self.num_frequencies * 2

    def forward(
            self,
            in_tensor: Float[Tensor, "*bs input_dim"],
            covs: Optional[Float[Tensor, "*bs input_dim input_dim"]] = None,
    ) -> Float[Tensor, "*bs output_dim"]:
        """Calculates RFF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.

        Returns:
            Output values will be between -1 and 1
        """
        in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        scaled_inputs = in_tensor @ self.b_matrix  # [..., "num_frequencies"]

        if covs is None:
            encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))
        else:
            input_var = torch.sum((covs @ self.b_matrix) * self.b_matrix, -2)
            encoded_inputs = expected_sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1), torch.cat(2 * [input_var], dim=-1)
            )

        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)

        return encoded_inputs


class HashEncoding(Encoding):
    """Hash encoding

    Args:
        num_levels: Number of feature grids.
        min_res: Resolution of smallest feature grid.
        max_res: Resolution of largest feature grid.
        log2_hashmap_size: Size of hash map is 2^log2_hashmap_size.
        features_per_level: Number of features per level.
        hash_init_scale: Value to initialize hash grid.
        implementation: Implementation of hash encoding. Fallback to torch if tcnn not available.
        interpolation: Interpolation override for tcnn hashgrid. Not supported for torch unless linear.
    """

    def __init__(
            self,
            num_levels: int = 16,
            min_res: int = 16,
            max_res: int = 1024,
            log2_hashmap_size: int = 19,
            features_per_level: int = 2,
            hash_init_scale: float = 0.001,
            implementation: Literal["tcnn", "torch"] = "tcnn",
            interpolation: Optional[Literal["Nearest", "Linear", "Smoothstep"]] = None,
    ) -> None:

        super().__init__(in_dim=3)
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hash_table_size = 2 ** log2_hashmap_size

        levels = torch.arange(num_levels)
        growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1))
        self.scalings = torch.floor(min_res * growth_factor ** levels)

        self.hash_offset = levels * self.hash_table_size
        self.hash_table = torch.rand(size=(self.hash_table_size * num_levels, features_per_level)) * 2 - 1
        self.hash_table *= hash_init_scale
        self.hash_table = nn.Parameter(self.hash_table)

        self.tcnn_encoding = None
        if not TCNN_EXISTS and implementation == "tcnn":
            logger.warning("tcnn not available, falling back to torch implementation")
        elif implementation == "tcnn":
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": self.num_levels,
                "n_features_per_level": self.features_per_level,
                "log2_hashmap_size": self.log2_hashmap_size,
                "base_resolution": min_res,
                "per_level_scale": growth_factor,
            }
            if interpolation is not None:
                encoding_config["interpolation"] = interpolation

            self.tcnn_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config,
            )

        if not TCNN_EXISTS or self.tcnn_encoding is None:
            assert (
                    interpolation is None or interpolation == "Linear"
            ), f"interpolation '{interpolation}' is not supported for torch encoding backend"

    def get_out_dim(self) -> int:
        return self.num_levels * self.features_per_level

    def hash_fn(self, in_tensor: Float[Tensor, "bs num_levels 3"]) -> Float[Tensor, "*bs num_levels"]:
        """Returns hash tensor using method described in Instant-NGP

        Args:
            in_tensor: Tensor to be hashed
        """

        # min_val = torch.min(in_tensor)
        # max_val = torch.max(in_tensor)
        # assert min_val >= 0.0
        # assert max_val <= 1.0

        in_tensor = in_tensor * torch.tensor([1, 2654435761, 805459861]).to(in_tensor.device)
        x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
        x = torch.bitwise_xor(x, in_tensor[..., 2])
        x %= self.hash_table_size
        x += self.hash_offset.to(x.device)
        return x

    def pytorch_fwd(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""

        assert in_tensor.shape[-1] == 3
        in_tensor = in_tensor[..., None, :]  # [..., 1, 3]
        scaled = in_tensor * self.scalings.view(-1, 1).to(in_tensor.device)  # [..., L, 3]
        scaled_c = torch.ceil(scaled).type(torch.int32)
        scaled_f = torch.floor(scaled).type(torch.int32)

        offset = scaled - scaled_f

        hashed_0 = self.hash_fn(scaled_c)  # [..., num_levels]
        hashed_1 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_2 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_3 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_4 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_5 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_6 = self.hash_fn(scaled_f)
        hashed_7 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))

        f_0 = self.hash_table[hashed_0]  # [..., num_levels, features_per_level]
        f_1 = self.hash_table[hashed_1]
        f_2 = self.hash_table[hashed_2]
        f_3 = self.hash_table[hashed_3]
        f_4 = self.hash_table[hashed_4]
        f_5 = self.hash_table[hashed_5]
        f_6 = self.hash_table[hashed_6]
        f_7 = self.hash_table[hashed_7]

        f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
        f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])
        f_56 = f_5 * offset[..., 0:1] + f_6 * (1 - offset[..., 0:1])
        f_47 = f_4 * offset[..., 0:1] + f_7 * (1 - offset[..., 0:1])

        f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])
        f4756 = f_47 * offset[..., 1:2] + f_56 * (1 - offset[..., 1:2])

        encoded_value = f0312 * offset[..., 2:3] + f4756 * (
                1 - offset[..., 2:3]
        )  # [..., num_levels, features_per_level]

        return torch.flatten(encoded_value, start_dim=-2, end_dim=-1)  # [..., num_levels * features_per_level]

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        if TCNN_EXISTS and self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)


def components_from_spherical_harmonics(
        levels: int, directions: Float[Tensor, "... 3"]
) -> Float[Tensor, "... components"]:
    """
    Returns value for each component of spherical harmonics.

    Args:
        levels: Number of spherical harmonic levels to compute.
        directions: Spherical harmonic coefficients
    """
    num_components = levels ** 2
    components = torch.zeros((*directions.shape[:-1], num_components), device=directions.device)

    assert 1 <= levels <= 5, f"SH levels must be in [1,4], got {levels}"
    assert directions.shape[-1] == 3, f"Direction input should have three dimensions. Got {directions.shape[-1]}"

    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]

    xx = x ** 2
    yy = y ** 2
    zz = z ** 2

    # l0
    components[..., 0] = 0.28209479177387814

    # l1
    if levels > 1:
        components[..., 1] = 0.4886025119029199 * y
        components[..., 2] = 0.4886025119029199 * z
        components[..., 3] = 0.4886025119029199 * x

    # l2
    if levels > 2:
        components[..., 4] = 1.0925484305920792 * x * y
        components[..., 5] = 1.0925484305920792 * y * z
        components[..., 6] = 0.9461746957575601 * zz - 0.31539156525251999
        components[..., 7] = 1.0925484305920792 * x * z
        components[..., 8] = 0.5462742152960396 * (xx - yy)

    # l3
    if levels > 3:
        components[..., 9] = 0.5900435899266435 * y * (3 * xx - yy)
        components[..., 10] = 2.890611442640554 * x * y * z
        components[..., 11] = 0.4570457994644658 * y * (5 * zz - 1)
        components[..., 12] = 0.3731763325901154 * z * (5 * zz - 3)
        components[..., 13] = 0.4570457994644658 * x * (5 * zz - 1)
        components[..., 14] = 1.445305721320277 * z * (xx - yy)
        components[..., 15] = 0.5900435899266435 * x * (xx - 3 * yy)

    # l4
    if levels > 4:
        components[..., 16] = 2.5033429417967046 * x * y * (xx - yy)
        components[..., 17] = 1.7701307697799304 * y * z * (3 * xx - yy)
        components[..., 18] = 0.9461746957575601 * x * y * (7 * zz - 1)
        components[..., 19] = 0.6690465435572892 * y * (7 * zz - 3)
        components[..., 20] = 0.10578554691520431 * (35 * zz * zz - 30 * zz + 3)
        components[..., 21] = 0.6690465435572892 * x * z * (7 * zz - 3)
        components[..., 22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1)
        components[..., 23] = 1.7701307697799304 * x * z * (xx - 3 * yy)
        components[..., 24] = 0.4425326924449826 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

    return components


class SHEncoding(Encoding):
    """Spherical harmonic encoding

    Args:
        levels: Number of spherical harmonic levels to encode.
    """

    def __init__(self, levels: int = 4) -> None:
        super().__init__(in_dim=3)

        if levels <= 0 or levels > 4:
            raise ValueError(f"Spherical harmonic encoding only supports 1 to 4 levels, requested {levels}")

        self.levels = levels

    def get_out_dim(self) -> int:
        return self.levels ** 2

    @torch.no_grad()
    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        return components_from_spherical_harmonics(levels=self.levels, directions=in_tensor)
