# Copied from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/cameras/rays.py

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
Some ray datastructures.
"""
import random
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
from jaxtyping import Float

from utils.tensor_dataclass import TensorDataclass


@dataclass
class Gaussians:
    """Stores Gaussians

    Args:
        mean: Mean of multivariate Gaussian
        cov: Covariance of multivariate Gaussian.
    """

    mean: Float[torch.Tensor, "... dim"]
    cov: Float[torch.Tensor, "... dim dim"]


def compute_3d_gaussian(
        directions: Float[torch.Tensor, "... 3"],
        means: Float[torch.Tensor, "... 3"],
        dir_variance: Float[torch.Tensor, "... 1"],
        radius_variance: Float[torch.Tensor, "... 1"],
) -> Gaussians:
    """Compute gaussian along ray.

    Args:
        directions: Axis of Gaussian.
        means: Mean of Gaussian.
        dir_variance: Variance along direction axis.
        radius_variance: Variance tangent to direction axis.

    Returns:
        Gaussians: Oriented 3D gaussian.
    """

    dir_outer_product = directions[..., :, None] * directions[..., None, :]
    eye = torch.eye(directions.shape[-1], device=directions.device)
    dir_mag_sq = torch.clamp(torch.sum(directions**2, dim=-1, keepdim=True), min=1e-10)
    null_outer_product = eye - directions[..., :, None] * (directions / dir_mag_sq)[..., None, :]
    dir_cov_diag = dir_variance[..., None] * dir_outer_product[..., :, :]
    radius_cov_diag = radius_variance[..., None] * null_outer_product[..., :, :]
    cov = dir_cov_diag + radius_cov_diag
    return Gaussians(mean=means, cov=cov)


def conical_frustum_to_gaussian(
        origins: Float[torch.Tensor, "... 3"],
        directions: Float[torch.Tensor, "... 3"],
        starts: Float[torch.Tensor, "... 1"],
        ends: Float[torch.Tensor, "... 1"],
        radius: Float[torch.Tensor, "... 1"],
) -> Gaussians:
    """Approximates conical frustums with a Gaussian distributions.

    Uses stable parameterization described in mip-NeRF publication.

    Args:
        origins: Origins of cones.
        directions: Direction (axis) of frustums.
        starts: Start of conical frustums.
        ends: End of conical frustums.
        radius: Radii of cone a distance of 1 from the origin.

    Returns:
        Gaussians: Approximation of conical frustums
    """
    mu = (starts + ends) / 2.0
    hw = (ends - starts) / 2.0
    means = origins + directions * (mu + (2.0 * mu * hw**2.0) / (3.0 * mu**2.0 + hw**2.0))
    dir_variance = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) / (3 * mu**2 + hw**2) ** 2)
    radius_variance = radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 * (hw**4) / (3 * mu**2 + hw**2))
    return compute_3d_gaussian(directions, means, dir_variance, radius_variance)


@dataclass
class Frustums(TensorDataclass):
    """Describes region of space as a frustum."""

    origins: Float[torch.Tensor, "*bs 3"]
    """xyz coordinate for ray origin."""
    directions: Float[torch.Tensor, "*bs 3"]
    """Direction of ray."""
    starts: Float[torch.Tensor, "*bs 1"]
    """Where the frustum starts along a ray."""
    ends: Float[torch.Tensor, "*bs 1"]
    """Where the frustum ends along a ray."""
    pixel_area: Float[torch.Tensor, "*bs 1"]
    """Projected area of pixel a distance 1 away from origin."""
    offsets: Optional[Float[torch.Tensor, "*bs 3"]] = None
    """Offsets for each sample position"""

    def get_positions(self) -> Float[torch.Tensor, "... 3"]:
        """Calculates "center" position of frustum. Not weighted by mass.

        Returns:
            xyz positions.
        """
        pos = self.origins + self.directions * (self.starts + self.ends) / 2
        if self.offsets is not None:
            pos = pos + self.offsets
        return pos

    def set_offsets(self, offsets):
        """Sets offsets for this frustum for computing positions"""
        self.offsets = offsets

    def get_gaussian_blob(self) -> Gaussians:
        """Calculates guassian approximation of conical frustum.

        Returns:
            Conical frustums approximated by gaussian distribution.
        """
        # Cone radius is set such that the square pixel_area matches the cone area.
        cone_radius = torch.sqrt(self.pixel_area) / 1.7724538509055159  # r = sqrt(pixel_area / pi)
        if self.offsets is not None:
            raise NotImplementedError()
        return conical_frustum_to_gaussian(
            origins=self.origins,
            directions=self.directions,
            starts=self.starts,
            ends=self.ends,
            radius=cone_radius,
        )

    @classmethod
    def get_mock_frustum(cls, device="cpu") -> "Frustums":
        """Helper function to generate a placeholder frustum.

        Returns:
            A size 1 frustum with meaningless values.
        """
        return Frustums(
            origins=torch.ones((1, 3)).to(device),
            directions=torch.ones((1, 3)).to(device),
            starts=torch.ones((1, 1)).to(device),
            ends=torch.ones((1, 1)).to(device),
            pixel_area=torch.ones((1, 1)).to(device),
        )


@dataclass
class RaySamples(TensorDataclass):
    """Samples along a ray"""

    frustums: Frustums
    """Frustums along ray."""
    camera_indices: Optional[Float[torch.Tensor, "*bs 1"]] = None
    """Camera index."""
    deltas: Optional[Float[torch.Tensor, "*bs 1"]] = None
    """"width" of each sample."""
    spacing_starts: Optional[Float[torch.Tensor, "*bs num_samples 1"]] = None
    """Start of normalized bin edges along ray [0,1], before warping is applied, ie. linear in disparity sampling."""
    spacing_ends: Optional[Float[torch.Tensor, "*bs num_samples 1"]] = None
    """Start of normalized bin edges along ray [0,1], before warping is applied, ie. linear in disparity sampling."""
    spacing_to_euclidean_fn: Optional[Callable] = None
    """Function to convert bins to euclidean distance."""
    metadata: Optional[Dict[str, Float[torch.Tensor, "*bs latent_dims"]]] = None
    """additional information relevant to generating ray samples"""

    pl_positions: Optional[Float[torch.Tensor, "... 1"]] = None
    """Point light positions with which rays are sampled"""

    def get_weights(self, densities: Float[torch.Tensor, "... num_samples 1"]) \
            -> Float[torch.Tensor, "... num_samples 1"]:
        """Return weights based on predicted densities

        Args:
            densities: Predicted densities for samples along ray

        Returns:
            Weights for each sample
        """

        delta_density = self.deltas * densities
        alphas = 1 - torch.exp(-delta_density)

        transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
        transmittance = torch.cat(
            [torch.zeros((*transmittance.shape[:1], 1, 1), device=densities.device), transmittance], dim=-2
        )
        transmittance = torch.exp(-transmittance)  # [..., "num_samples"]

        weights = alphas * transmittance  # [..., "num_samples"]
        weights = torch.nan_to_num(weights)

        return weights


@dataclass
class RayBundle(TensorDataclass):
    """A bundle of ray parameters."""

    origins: Float[torch.Tensor, "num_rays 3"]
    """Ray origins (XYZ)"""
    directions: Float[torch.Tensor, "num_rays 3"]
    """Unit ray direction vector"""
    pl_positions: Float[torch.Tensor, "num_rays 1"]
    """Point light positions with which rays are sampled"""

    camera_indices: Optional[Float[torch.Tensor, "num_rays 1"]] = None
    """Camera indices"""

    pixel_area: Optional[Float[torch.Tensor, "num_rays 1"]] = None
    """Projected area of pixel a distance 1 away from origin"""
    nears: Optional[Float[torch.Tensor, "num_rays 1"]] = None
    """Distance along ray to start sampling"""
    fars: Optional[Float[torch.Tensor, "num_rays 1"]] = None
    """Rays Distance along ray to stop sampling"""
    metadata: Optional[Dict[str, Float[torch.Tensor, "num_rays latent_dims"]]] = None
    """Additional metadata or data needed for interpolation, will mimic shape of rays"""

    def set_camera_indices(self, camera_index: int) -> None:
        """Sets all the camera indices to a specific camera index.

        Args:
            camera_index: Camera index.
        """
        self.camera_indices = torch.ones_like(self.origins[..., 0:1]).long() * camera_index

    def __len__(self):
        num_rays = torch.numel(self.origins) // self.origins.shape[-1]
        return num_rays

    def sample(self, num_rays: int) -> "RayBundle":
        """Returns a RayBundle as a subset of rays.

        Args:
            num_rays: Number of rays in output RayBundle

        Returns:
            RayBundle with subset of rays.
        """
        assert num_rays <= len(self)
        indices = random.sample(range(len(self)), k=num_rays)
        return self[indices]

    def get_row_major_sliced_ray_bundle(self, start_idx: int, end_idx: int) -> "RayBundle":
        """Flattens RayBundle and extracts chunk given start and end indices.

        Args:
            start_idx: Start index of RayBundle chunk.
            end_idx: End index of RayBundle chunk.

        Returns:
            Flattened RayBundle with end_idx-start_idx rays.

        """
        return self.flatten()[start_idx:end_idx]

    def get_ray_samples(
            self,
            bin_starts: Float[torch.Tensor, "*bs num_samples 1"],
            bin_ends: Float[torch.Tensor, "*bs num_samples 1"],
            spacing_starts: Optional[Float[torch.Tensor, "*bs num_samples 1"]] = None,
            spacing_ends: Optional[Float[torch.Tensor, "*bs num_samples 1"]] = None,
            spacing_to_euclidean_fn: Optional[Callable] = None,
    ) -> RaySamples:
        """Produces samples for each ray by projection points along the ray direction. Currently, samples uniformly.

        Args:
            bin_starts: Distance from origin to start of bin.
            bin_ends: Distance from origin to end of bin.
            spacing_starts:
            spacing_ends:
            spacing_to_euclidean_fn:

        Returns:
            Samples projected along ray.
        """
        deltas = bin_ends - bin_starts
        if self.camera_indices is not None:
            camera_indices = self.camera_indices[..., None]
        else:
            camera_indices = None

        shaped_raybundle_fields = self[..., None]

        frustums = Frustums(
            origins=shaped_raybundle_fields.origins,  # [..., 1, 3]
            directions=shaped_raybundle_fields.directions,  # [..., 1, 3]
            starts=bin_starts,  # [..., num_samples, 1]
            ends=bin_ends,  # [..., num_samples, 1]
            pixel_area=shaped_raybundle_fields.pixel_area,  # [..., 1, 1]
        )

        ray_samples = RaySamples(
            frustums=frustums,
            camera_indices=camera_indices,  # [..., 1, 1]
            deltas=deltas,  # [..., num_samples, 1]
            spacing_starts=spacing_starts,  # [..., num_samples, 1]
            spacing_ends=spacing_ends,  # [..., num_samples, 1]
            spacing_to_euclidean_fn=spacing_to_euclidean_fn,
            metadata=shaped_raybundle_fields.metadata,
            pl_positions=None if self.pl_positions is None else self.pl_positions[..., None],  # [..., 1, 1]
        )

        return ray_samples


if __name__ == '__main__':
    ray_bundle = RayBundle(
        origins=torch.rand(10, 3),
        directions=torch.rand(10, 3),
        camera_indices=torch.rand(10, 1),
        nears=torch.rand(10, 1),
        fars=torch.rand(10, 1),
        pl_positions=torch.rand(10, 1),
    )
    print(ray_bundle.shape)
