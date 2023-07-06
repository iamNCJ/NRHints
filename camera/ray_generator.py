from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from camera.camera_model import CameraModel
from camera.lie_groups import exp_map_SO3xR3, exp_map_SE3
from camera.ray_utils import RayBundle
from data.data_loader import RawPixelBundle


@dataclass(frozen=True)
class RayGeneratorConfig:
    """Configuration of ray generator."""

    override_near_far_from_sphere: bool = True
    """Whether to override near and far plane with the sphere radius. Used by NeuS."""

    cam_opt_mode: Literal["off", "SO3xR3", "SE3"] = "off"
    """Camera pose optimization strategy to use. If enabled, we recommend SO3xR3."""

    pl_opt: bool = False
    """Whether to optimize point light positions."""

    opt_lr: float = 3e-5
    """Learning rate for the camera & point light optimizer."""

    cam_position_noise_std: float = 0.0
    """Noise to add to initial positions. Useful for synthetic experiments."""

    cam_orientation_noise_std: float = 0.0
    """Noise to add to initial orientations. Useful for synthetic experiments."""

    pl_position_noise_std: float = 0.0
    """Noise to add to initial point light positions. Useful for synthetic experiments."""


class RayGenerator(nn.Module):
    """
    Generate rays from camera model and applies refinement according to image indices.
    """
    def __init__(self, camera: CameraModel, num_cameras: int, config: RayGeneratorConfig):
        super().__init__()
        self.camera = camera
        self.config = config

        # Initialize learnable parameters.
        if self.config.cam_opt_mode == "off":
            pass
        elif self.config.cam_opt_mode in ("SO3xR3", "SE3"):
            self.cam_pose_adjustment = torch.nn.Parameter(torch.zeros((num_cameras, 6)))
        else:
            raise ValueError(f"Unknown camera pose optimization mode: {self.config.cam_opt_mode}")

        if self.config.pl_opt:
            self.pl_adjustment = torch.nn.Parameter(torch.zeros((num_cameras, 3)))

        # Initialize camera pose noise.
        if self.config.cam_position_noise_std != 0.0 or self.config.cam_orientation_noise_std != 0.0:
            assert self.config.cam_position_noise_std >= 0.0 and self.config.cam_orientation_noise_std >= 0.0
            std_vector = torch.tensor([
                [self.config.cam_position_noise_std] * 3 + [self.config.cam_orientation_noise_std] * 3
            ], dtype=torch.float32)
            cam_pose_noise = exp_map_SE3(torch.normal(torch.zeros((num_cameras, 6)), std_vector))
            self.register_buffer("cam_pose_noise", cam_pose_noise, persistent=True)

        # Initialize point light noise.
        if self.config.pl_position_noise_std != 0.0:
            assert self.config.pl_position_noise_std >= 0.0
            pl_noise = torch.normal(torch.zeros((num_cameras, 3)), self.config.pl_position_noise_std)
            self.register_buffer("pl_noise", pl_noise, persistent=True)

    def forward(
            self,
            pixel_bundle: RawPixelBundle,
    ) -> RayBundle:
        x = pixel_bundle.w_indices[..., 0] + 0.5
        y = pixel_bundle.h_indices[..., 0] + 0.5
        if pixel_bundle.img_indices is not None:
            img_indices = pixel_bundle.img_indices[..., 0]
        else:
            img_indices = None
        dirs = torch.stack([
            (x - self.camera.cx) / self.camera.fx,
            -(y - self.camera.cy) / self.camera.fy,
            -torch.ones_like(x)
        ], dim=-1)
        R = pixel_bundle.poses[:, :3, :3]
        t = pixel_bundle.poses[:, :3, 3:]

        # Apply initial camera pose noise.
        if hasattr(self, "cam_pose_noise") and img_indices is not None:
            dR = self.cam_pose_noise[img_indices, :3, :3]
            dt = self.cam_pose_noise[img_indices, :3, 3:]
            R = dR.matmul(R)
            t = dt + dR.matmul(t)

        # Apply learned camera transformation delta.
        cam_opt_mat = None
        if self.config.cam_opt_mode == "off" or img_indices is None:
            # video views don't have valid img_indices / rgb_gt, and should apply cam_opt
            pass
        elif self.config.cam_opt_mode == "SO3xR3":
            cam_opt_mat = exp_map_SO3xR3(self.cam_pose_adjustment[img_indices, :])
        elif self.config.cam_opt_mode == "SE3":
            cam_opt_mat = exp_map_SE3(self.cam_pose_adjustment[img_indices, :])
        else:
            raise ValueError(f"Unknown camera pose optimization mode: {self.config.cam_opt_mode}")

        if cam_opt_mat is not None:
            dR = cam_opt_mat[:, :3, :3]
            dt = cam_opt_mat[:, :3, 3:]
            R = dR.matmul(R)
            t = dt + dR.matmul(t)

        pls = pixel_bundle.pls

        # Apply initial point light noise.
        if hasattr(self, "pl_noise") and img_indices is not None:
            pls = pls + self.pl_noise[img_indices, :]

        # Apply learned point light delta.
        if self.config.pl_opt and pixel_bundle.img_indices is not None:
            pls = pls + self.pl_adjustment[img_indices, :]

        rays_d = torch.sum(dirs[..., None, :] * R, dim=-1)
        rays_d = F.normalize(rays_d, dim=-1, p=2)
        rays_o = t[..., 0]
        rays_pls = pls

        if self.config.override_near_far_from_sphere:
            # Override near and far plane with the sphere radius.
            a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
            b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
            mid = 0.5 * (-b) / a
            near = mid - 1.0
            far = mid + 1.0
        else:
            near = self.camera.zn * torch.ones_like(rays_o[..., :1])
            far = self.camera.zf * torch.ones_like(rays_o[..., :1])

        return RayBundle(
            origins=rays_o,
            directions=rays_d,
            pl_positions=rays_pls,
            nears=near,
            fars=far,
        )


if __name__ == '__main__':
    H = 512
    W = 1024
    BS = 16
    CAM_NUM = 700
    h = torch.randint(0, H, (BS, 1))
    w = torch.randint(0, W, (BS, 1))
    i = torch.randint(0, CAM_NUM, (BS, 1))
    print(h.shape, w.shape)
    c2w = torch.rand((BS, 4, 4))
    pls = torch.rand((BS, 3))
    camera = CameraModel(H, W, W // 2, H // 2, W // 2, H // 2, 0, 1)
    ray_generator = RayGenerator(camera, CAM_NUM, RayGeneratorConfig())
    ray_bundle = ray_generator(
        RawPixelBundle(
            img_indices=i,
            h_indices=h,
            w_indices=w,
            poses=c2w,
            pls=pls,
            rgb_gt=torch.rand((BS, 3))
        )
    )
    print(ray_bundle.shape)
    print(ray_bundle[0].nears.shape)
