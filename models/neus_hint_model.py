from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List

import mcubes
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from camera.ray_utils import RayBundle
from fields.nerf_density_field import NeRF, NeRFConfig
from fields.reflectance_network import ReflectanceNetwork, ReflectanceNetConfig
from fields.sdf_field import SDFNetwork, SDFNetConfig
from utils.tensor_dataclass import TensorDataclass


# Hierarchical sampling
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    device = weights.device
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1], device=device), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.tensor(u, device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1, device=device), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds, device=device), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom, device=device), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


@dataclass(frozen=True)
class SingleVarianceNetConfig:
    """Configuration for the single variance network."""

    init_val: float = 0.3
    """Initial value for the variance (actually 1/s)."""


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1], device=x.device) * torch.exp(self.variance * 10.0)


class DepthComputationType(Enum):
    """Methods to compute depth."""

    AlphaBlend = 'alpha_blending'
    """Alpha blending a depth according to weights."""
    MaximalWeightPoint = 'maximum_point'
    """Use the point with maximal weight as depth, following NeuS's unbiasedness property."""
    SphereTracing = 'sphere_tracing'
    """Use sphere tracing to compute depth, slower."""


class NormalComputationType(Enum):
    """Methods to compute normal."""

    Analytic = 'analytic'
    """Analytic normal from derivative of sdf."""
    NormalizedAnalytic = 'normalized_analytic'
    """Analytic normal from derivative of sdf with normalization."""


@dataclass(frozen=True)
class NeuSRendererConfig:
    """Configuration for the NeuS renderer."""

    use_outside_nerf: bool = False
    """Whether to use outside nerf for background."""
    n_samples: int = 64
    """Number of stratified samples per ray."""
    n_importance_samples: int = 64
    """Number of importance samples per ray."""
    n_outside_samples: int = 32
    """Number of samples per ray for outside nerf."""
    normal_type: NormalComputationType = NormalComputationType.NormalizedAnalytic
    """Method to compute normal."""
    up_sample_steps: int = 4
    """Number of steps to up-sample during hierarchical sampling."""
    depth_type: DepthComputationType = DepthComputationType.AlphaBlend
    """Method to compute depth."""
    shadow_hint: bool = True
    """Whether to use shadow hint."""
    force_shadow_map: bool = False
    """Whether to force dumping a shadow map, even if shadow hint is not used."""
    specular_hint: bool = True
    """Whether to use specular hint."""
    force_specular_cue: bool = False
    """Whether to force dumping a specular cue, even if specular hint is not used."""
    shadow_ray_offset: float = 1e-2
    """Offset of shadow ray for shadow map calculation."""
    specular_roughness: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.13, 0.34])
    """Specular roughness for specular cue calculation."""
    shadow_hint_gradient: bool = False
    """Whether to use gradient of shadow hint."""
    specular_hint_gradient: bool = False
    """Whether to use gradient of specular hint."""
    n_shadow_importance_clip: int = -1
    """Number of samples to calculate for shadow hint, -1 means only use estimated hit point calculated by depth."""
    n_shadow_samples: int = 64
    """Number of samples to calculate for shadow hint."""
    n_shadow_importance_samples: int = 64
    """Number of importance samples to calculate for shadow hint."""
    override_near_far_to_sphere: bool = True
    """Whether to override near and far plane from unit sphere."""


@dataclass(frozen=True)
class NeuSModelConfig:
    """Configuration for the whole NeuS Hint Relighting model."""

    sdf_network: SDFNetConfig = SDFNetConfig()
    outside_nerf: NeRFConfig = NeRFConfig()
    deviation_network: SingleVarianceNetConfig = SingleVarianceNetConfig()
    reflectance_network: ReflectanceNetConfig = ReflectanceNetConfig()
    renderer: NeuSRendererConfig = NeuSRendererConfig()

    # loss related
    igr_weight: float = 0.1
    """Weight for the igr / eikonal loss."""

    # lr scheduling related
    lr: float = 5e-4
    """Learning rate."""
    lr_alpha: float = 0.05
    """Learning rate hp: alpha."""
    warm_up_end: int = 5_000
    """Number of steps to warm up."""
    end_iter: int = 1_000_000
    """Number of steps to train."""
    anneal_end: int = 50_000
    """Number of steps to anneal."""
    geometry_warmup_end: int = 0
    """Number of steps to warm up geometry, during warm up, all hints are set to 0."""

    # chunk sizes
    batch_size: int = 512
    """Batch size."""
    shadow_mini_chunk_size: int = 2048
    """Mini chunk size for batched shadow hint calculation."""
    training_chunk_size: int = 512
    """Chunk size for training."""
    inference_chunk_size: int = 512
    """Chunk size for inference (testing)."""


@dataclass
class RenderOutput(TensorDataclass):
    rgb: Float[torch.Tensor, "bs 3"]
    depth: Float[torch.Tensor, "bs 1"]
    weights: Float[torch.Tensor, "bs n_samples"]
    s_val: Float[torch.Tensor, "bs 1"]
    inside_sphere: Float[torch.Tensor, "bs n_samples"]
    relax_inside_sphere: Float[torch.Tensor, "bs n_samples"]
    analytic_normals: Float[torch.Tensor, "bs n_samples 3"]
    normalized_analytic_normals: Float[torch.Tensor, "bs n_samples 3"]
    visibilities: Optional[Float[torch.Tensor, "bs 1"]] = None
    specular_cue: Optional[Float[torch.Tensor, "bs n_samples n_roughnesses"]] = None

    _field_custom_dimensions = {
        "analytic_normals": 2,
        "normalized_analytic_normals": 2,
        "specular_cue": 2,
    }


class NeuSHintRenderer(nn.Module):
    def __init__(self, config: NeuSModelConfig):
        super().__init__()
        self.has_shadow_hint = config.renderer.shadow_hint or config.renderer.force_shadow_map
        self.has_specular_hint = config.renderer.specular_hint or config.renderer.force_specular_cue

        self.sdf_network = SDFNetwork(config.sdf_network)
        self.deviation_network = SingleVarianceNetwork(
            init_val=config.deviation_network.init_val
        )
        color_d_in = 12
        if self.has_shadow_hint:
            color_d_in += 1
        if self.has_specular_hint:
            color_d_in += len(config.renderer.specular_roughness)
        self.color_network = ReflectanceNetwork(
            d_feature=config.sdf_network.d_out_feat,
            d_in=color_d_in,
            d_out=3,  # RGB
            config=config.reflectance_network,
            shadow_hint=config.renderer.shadow_hint,
            specular_hint=config.renderer.specular_hint,
            specular_hint_len=len(config.renderer.specular_roughness)
        )
        self.has_outside_nerf = config.renderer.use_outside_nerf
        if self.has_outside_nerf:
            self.outside_nerf = NeRF(
                d_in=4,  # normalized x, y, z position & distance to center
                d_in_view=6,  # view dir & pl pos
                config=config.outside_nerf,
            )
        self.config = config

    @staticmethod
    def up_sample(rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        device = rays_o.device
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1], device=device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def get_alpha(self, pts, dists, dirs, cos_anneal_ratio=1.):
        device = pts.device
        sdf = self.sdf_network.sdf(pts)
        gradients = self.sdf_network.gradient(pts).squeeze()
        inv_s = self.deviation_network(torch.zeros([1, 3], device=device))[:, :1].clip(1e-6, 1e6)  # Single parameter
        inv_s = inv_s.expand(sdf.size(0), 1)
        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0, 1)
        return alpha, c, gradients, inv_s, alpha, None, None

    @torch.no_grad()
    def sphere_trace(self, rays_o, rays_d, num_iterations, convergence_threshold, far):
        device = rays_o.device
        pts = rays_o
        depths = torch.zeros((pts.size(0), 1), device=device)
        for _ in range(num_iterations):
            sdf = self.sdf_network.sdf(pts)  # [N, 1]
            converged = (torch.abs(sdf) < convergence_threshold) | (depths > far)  # [N, 1]
            pts = torch.where(converged, pts, pts + sdf * rays_d)  # [N, 3]
            depths = torch.where(converged, depths, depths + sdf)  # [N, 1]
            if converged.all():
                break
        return pts, depths

    def get_visibility(self, pls, target_points, up_sample_steps=4,
                       cos_anneal_ratio=1.0,
                       offset=1e-2, perturb=False):
        device = pls.device
        n_samples = self.config.renderer.n_shadow_samples
        n_importance_samples = self.config.renderer.n_shadow_importance_samples
        with nullcontext() if self.config.renderer.shadow_hint_gradient else torch.no_grad():
            shadow_ray_o = pls
            shadow_ray_d = target_points - shadow_ray_o
            light_norms = torch.linalg.norm(shadow_ray_d, ord=2, dim=-1, keepdim=True)
            sample_dist = light_norms / n_samples
            shadow_ray_d = shadow_ray_d / light_norms
            z_vals = torch.linspace(0., 1., steps=n_samples, device=device)
            z_vals = z_vals * light_norms * (1. - offset)

            if perturb:
                # get intervals between samples
                mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
                upper = torch.cat([mids, z_vals[..., -1:]], -1)
                lower = torch.cat([z_vals[..., :1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand(z_vals.shape, device=device)
                z_vals = lower + (upper - lower) * t_rand
            # pts_alpha = pls[..., None, None,:] - z_vals[...,None] * shadow_ray_d[..., None,:]
            if n_importance_samples > 0:
                pts = shadow_ray_o[:, None, :] + shadow_ray_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(-1, n_samples)
                for i in range(up_sample_steps):
                    new_z_vals = self.up_sample(shadow_ray_o,
                                                shadow_ray_d,
                                                z_vals,
                                                sdf,
                                                n_importance_samples // up_sample_steps,
                                                64 * 2 ** i)
                    z_vals, sdf = self.cat_z_vals(shadow_ray_o,
                                                  shadow_ray_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == up_sample_steps))
            batch_size, n_samples = z_vals.shape

            # Section length
            dists = z_vals[..., 1:] - z_vals[..., :-1]
            dists = torch.cat([dists, sample_dist.expand(dists[..., :1].shape)], -1)
            mid_z_vals = z_vals + dists * 0.5

            # Section midpoints
            pts = shadow_ray_o[:, None, :] + shadow_ray_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
            dirs = shadow_ray_d[:, None, :].expand(pts.shape)

            pts = pts.reshape(-1, 3)
            dirs = dirs.reshape(-1, 3)

            alpha, _, _, _, _, _, _ = self.get_alpha(pts, dists, dirs, cos_anneal_ratio)
            alpha = alpha.reshape(batch_size, n_samples)
            taus = torch.cumprod(
                torch.cat([torch.ones([batch_size, 1], device=device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        return taus[..., -1:]  # [N, 1]

    def render_outside(self, rays_o, rays_d, rays_pl, z_vals, sample_dist, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)  # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)
        pls = rays_pl[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 4)
        dirs = dirs.reshape(-1, 3)
        pls = pls.reshape(-1, 3)

        density, sampled_color = self.outside_nerf(pts, dirs, pls)
        sampled_color = torch.sigmoid(sampled_color)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def render_core(self,
                    rays_o,
                    rays_d,
                    rays_pl,
                    z_vals,
                    sample_dist,
                    background_rgb=None,
                    cos_anneal_ratio=1.0,
                    perturb_on_visibility=False,
                    background_alpha=None,
                    background_sampled_color=None,
                    geometry_warmup_state=False):
        device = rays_o.device
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([sample_dist], device=device).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)
        pls = rays_pl[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)
        pls = pls.reshape(-1, 3)

        sdf_nn_output = self.sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:1 + self.config.sdf_network.d_out_feat]

        alpha, cdf, gradients, inv_s, neus_alpha, res_alpha, res_feat = self.get_alpha(pts, dists, dirs,
                                                                                       cos_anneal_ratio)
        alpha = alpha.reshape(batch_size, n_samples)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        # Append background alpha
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1], device=device), 1. - alpha + 1e-7], -1), -1
        )[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)
        neus_weights = weights[:, :n_samples]

        # Calculate depth map using NeuS
        if self.config.renderer.depth_type == DepthComputationType.SphereTracing:
            hit_points, depths = self.sphere_trace(rays_o, rays_d, 2000, 1e-4, 100)
        elif self.config.renderer.depth_type == DepthComputationType.AlphaBlend:
            with torch.no_grad():
                depths = (mid_z_vals[..., :, None] * neus_weights[:, :, None]).sum(dim=1)
                hit_points = rays_o + rays_d * depths
        elif self.config.renderer.depth_type == DepthComputationType.MaximalWeightPoint:
            with torch.no_grad():
                maximum_idx = torch.argmax(neus_weights, dim=1, keepdim=True)
                depths = torch.gather(mid_z_vals, 1, maximum_idx)
                hit_points = rays_o + rays_d * depths

        # Visibility hint, 0 for visible, 1 for shadowed
        visibilities = None  # per point
        shadow_map = None  # per pixel
        if self.has_shadow_hint and not geometry_warmup_state:
            n_shadow_importance_clip = self.config.renderer.n_shadow_importance_clip
            if n_shadow_importance_clip == -1:  # -1 means hit point
                visibilities = self.get_visibility(
                    rays_pl, hit_points,
                    cos_anneal_ratio=cos_anneal_ratio,
                    offset=self.config.renderer.shadow_ray_offset,
                    perturb=perturb_on_visibility
                )
                shadow_map = visibilities
                visibilities = visibilities[:, None, :].repeat(1, n_samples, 1).reshape(-1, 1)
            elif n_shadow_importance_clip > 0:  # partial visibility hint solution
                clip_ratio = n_samples // n_shadow_importance_clip
                vis_hint_z_vals = z_vals[:, torch.arange(0, z_vals.size(1), clip_ratio)]
                vis_hint_pts = (rays_o[..., None, :] + rays_d[..., None, :] * vis_hint_z_vals[..., :, None]).reshape(
                    (-1, 3))  # [(N_rays * N_shadow_importance_clip), 3]
                vis_hint_pls = rays_pl[:, None].repeat(1, n_shadow_importance_clip, 1).reshape((-1, 3))
                mini_bs = self.config.shadow_mini_chunk_size
                visibilities = []
                for i in range(0, batch_size * n_shadow_importance_clip, mini_bs):
                    visibilities.append(self.get_visibility(
                        vis_hint_pls[i:i + mini_bs].reshape((-1, 3)),
                        vis_hint_pts[i:i + mini_bs].reshape((-1, 3)),
                        cos_anneal_ratio=cos_anneal_ratio,
                        offset=self.config.renderer.shadow_ray_offset,
                        perturb=perturb_on_visibility
                    ))
                visibilities = torch.stack(visibilities) \
                    .reshape((-1, n_shadow_importance_clip, 1)) \
                    .repeat_interleave(clip_ratio, dim=1)  # [N_rays, N_shadow_importance_clip, 1]
                # pick maximal weight points as hit points to dump
                maximum_idx = torch.argmax(weights, dim=1, keepdim=True)
                shadow_map = torch.gather(visibilities[..., 0], 1, maximum_idx)
                visibilities = visibilities.reshape(-1, 1)
        elif self.has_shadow_hint and geometry_warmup_state:
            visibilities = torch.zeros((batch_size * n_samples, 1), device=device)
            shadow_map = visibilities[:batch_size]

        # Specular Cue
        specular_cue = None  # per point
        analytic_normal = gradients
        normalized_normal = F.normalize(analytic_normal, dim=-1, p=2)
        shading_normal = normalized_normal
        hit_point_normal = (shading_normal.reshape(batch_size, n_samples, 3) * weights[:, :n_samples, None]).sum(dim=1)
        hit_point_normal = F.normalize(hit_point_normal, dim=-1, p=2)
        if self.has_specular_hint and not geometry_warmup_state:
            with nullcontext() if self.config.renderer.specular_hint_gradient else torch.no_grad():
                # l, v, h
                lit_dirs = F.normalize(rays_pl - hit_points, dim=-1, p=2)
                view_dirs = F.normalize(-rays_d, dim=-1, p=2)
                half_vecs = F.normalize(lit_dirs + view_dirs, dim=-1, p=2)
                # dots
                n_dot_l = torch.sum(hit_point_normal * lit_dirs, dim=-1).clip(0., 1.)
                n_dot_v = torch.sum(hit_point_normal * view_dirs, dim=-1).clip(0., 1.)
                n_dot_h = torch.sum(hit_point_normal * half_vecs, dim=-1).clip(0., 1.)
                h_dot_v = torch.sum(half_vecs * view_dirs, dim=-1).clip(0., 1.)
                n_dot_h_2 = torch.pow(n_dot_h, 2)

                specular_cue = []
                for roughness in self.config.renderer.specular_roughness:
                    # G
                    k = (roughness + 1.) * (roughness + 1.) / 8.
                    g1 = n_dot_v / (n_dot_v * (1. - k) + k)
                    g2 = n_dot_l / (n_dot_l * (1. - k) + k)
                    g = g1 * g2
                    # N
                    a2 = roughness * roughness
                    ndf = a2 / (torch.pi * torch.pow((n_dot_h_2 * (a2 - 1.) + 1.), 2))
                    # F
                    f = 0.04 + 0.96 * torch.pow(1. - h_dot_v, 5)
                    cook_torrance_specular = ndf * g * f / (4. * n_dot_v + 1e-3)
                    specular_cue.append(cook_torrance_specular)
                specular_cue = torch.stack(specular_cue, dim=-1)
                specular_cue = specular_cue[:, None].repeat(1, n_samples, 1).reshape((batch_size * n_samples, -1))
        elif self.has_specular_hint and geometry_warmup_state:
            specular_cue = torch.zeros((batch_size * n_samples, len(self.config.renderer.specular_roughness)),
                                       device=device)

        input_normal = None
        if self.config.renderer.normal_type == NormalComputationType.Analytic:
            input_normal = analytic_normal
        elif self.config.renderer.normal_type == NormalComputationType.NormalizedAnalytic:
            input_normal = normalized_normal
        sampled_color = self.color_network(pts, input_normal, dirs, feature_vector, pls,
                                           visibilities, specular_cue).reshape(batch_size, n_samples, 3)

        # Blend with background color
        if background_alpha is not None:
            sampled_color = sampled_color * inside_sphere[:, :, None] + \
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:  # Fixed background
            color = color + background_rgb * (1.0 - weights_sum)

        return {
            'color': color,
            'sdf': sdf,
            'analytic_normals': analytic_normal.reshape(batch_size, n_samples, 3),
            'normalized_analytic_normals': normalized_normal.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s.reshape(batch_size, n_samples),
            'weights': weights,
            'inside_sphere': inside_sphere,
            'relax_inside_sphere': relax_inside_sphere,
            'depths': depths,
            'visibilities': shadow_map,
            'specular_cue': specular_cue.reshape(batch_size, n_samples, -1) if self.has_specular_hint else None
        }

    def forward(self, ray_bundle: RayBundle, is_training=False, background_rgb=None, global_step=0) -> RenderOutput:
        rays_o = ray_bundle.origins
        rays_d = ray_bundle.directions
        rays_pl = ray_bundle.pl_positions
        near = ray_bundle.nears
        far = ray_bundle.fars

        # Extract configs
        device = rays_o.device
        batch_size = len(rays_o)
        n_samples = self.config.renderer.n_samples
        n_importance = self.config.renderer.n_importance_samples
        n_outside = self.config.renderer.n_outside_samples
        up_sample_steps = self.config.renderer.up_sample_steps
        perturb = is_training
        geometry_warmup_state = is_training and global_step < self.config.geometry_warmup_end
        cos_anneal_ratio = 1.0  # default cosine annealing ratio should be 1.0
        if is_training and self.config.anneal_end > 0:
            cos_anneal_ratio = min([1.0, global_step / self.config.anneal_end])

        sample_dist = 2.0 / n_samples  # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, n_samples, device=device)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.has_outside_nerf:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (n_outside + 1.0), n_outside)

        if perturb:
            t_rand = (torch.rand([batch_size, 1], device=device) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / n_samples

            if self.has_outside_nerf:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.has_outside_nerf:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / n_samples

        # Up sample
        if n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_samples)

                for i in range(up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                n_importance // up_sample_steps,
                                                64 * 2 ** i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == up_sample_steps))

        # Background model
        background_alpha = None
        background_sampled_color = None
        if self.has_outside_nerf:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_outside(rays_o, rays_d, rays_pl, z_vals_feed, sample_dist)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    rays_pl,
                                    z_vals,
                                    sample_dist,
                                    background_rgb=background_rgb,
                                    cos_anneal_ratio=cos_anneal_ratio,
                                    perturb_on_visibility=perturb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    geometry_warmup_state=geometry_warmup_state)

        result = RenderOutput(
            rgb=ret_fine['color'],
            depth=ret_fine['depths'],
            weights=ret_fine['weights'],
            s_val=ret_fine['s_val'],
            inside_sphere=ret_fine['inside_sphere'],
            relax_inside_sphere=ret_fine['inside_sphere'],
            analytic_normals=ret_fine['analytic_normals'],
            normalized_analytic_normals=ret_fine['normalized_analytic_normals'],
            visibilities=ret_fine['visibilities'] if self.has_shadow_hint else None,
            specular_cue=ret_fine['specular_cue'] if self.has_specular_hint else None,
        )
        return result

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))
