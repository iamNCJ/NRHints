from typing import Tuple, Dict, List, Iterator, Union

import numpy as np
import torch
from torch import nn

from camera.ray_generator import RayGenerator
from configs.main_config import SystemConfig
from data.data_loader import RawPixelBundle
from data.shm_helper import NRDataSHMInfo
from models.neus_hint_model import NeuSHintRenderer, RenderOutput
from utils.metrics import PSNR, SSIM, LPIPS
from utils.tensor_dataclass import td_concat


class BaseNRHintPipeline(nn.Module):
    """
    Pipeline wrapper model for training and testing NRHint model.
    With this wrapper, we can easily save the whole model into ckpts and wrap into a DDP model.
    """

    def __init__(self, config: SystemConfig, shm_info: NRDataSHMInfo):
        super().__init__()
        self.config = config
        self.ray_generator = RayGenerator(
            camera=shm_info.camera,
            num_cameras=shm_info.total_image_num,
            config=config.ray_generator
        )
        self.renderer = NeuSHintRenderer(config.model)

    def get_param_groups(self) -> List[Dict[str, Union[Iterator[torch.nn.Parameter], float]]]:
        """Return parameter groups with hyper params."""
        return [
            {'params': self.renderer.parameters(), 'lr': self.config.model.lr},
            {'params': self.ray_generator.parameters(), 'lr': self.config.ray_generator.opt_lr}
        ]

    def forward(self, pixel_bundle: RawPixelBundle, global_step: int = 0) -> RenderOutput:
        ray_bundle = self.ray_generator.forward(pixel_bundle)
        rendering_res = self.renderer.forward(
            ray_bundle,
            background_rgb=torch.ones([1, 3], device=pixel_bundle.pls.device) if self.config.data.white_background
            else torch.zeros([1, 3], device=pixel_bundle.pls.device),
            is_training=True,
            global_step=global_step
        )
        return rendering_res

    def get_train_loss_dict(self, rendering_res: RenderOutput, pixel_bundle: RawPixelBundle) -> dict:
        """
        Compute training loss and other metrics, return them in a dict.
        :param rendering_res: RenderOutput, rendering result
        :param pixel_bundle: RawPixelBundle, input data
        :return: dict, loss dict
        """
        rgb_loss = nn.functional.l1_loss(rendering_res.rgb, pixel_bundle.rgb_gt, reduction='sum') / \
                   (rendering_res.rgb.size(0) + 1e-5)
        gradient_error = (torch.linalg.norm(rendering_res.analytic_normals, ord=2, dim=-1) - 1.0) ** 2
        relax_inside_sphere = rendering_res.relax_inside_sphere
        eikonal_loss = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)
        loss = rgb_loss + eikonal_loss * self.config.model.igr_weight
        return {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            's_val': rendering_res.s_val.mean(),
            'psnr': PSNR(rendering_res.rgb, pixel_bundle.rgb_gt),
        }

    @torch.enable_grad()
    @torch.inference_mode(False)
    def register_view(self, img_pixel_bundle, device, steps: int = 500):
        optimizer = torch.optim.Adam(self.ray_generator.parameters(), lr=self.config.ray_generator.opt_lr)
        for i in range(steps):
            h_indices = torch.randint(0, img_pixel_bundle.shape[0], (self.config.model.batch_size,), device='cpu')
            w_indices = torch.randint(0, img_pixel_bundle.shape[1], (self.config.model.batch_size,), device='cpu')
            pixel_bundle = img_pixel_bundle[h_indices, w_indices].to(device)
            ray_bundle = self.ray_generator.forward(pixel_bundle)
            rendering_res = self.renderer.forward(
                ray_bundle,
                background_rgb=torch.ones([1, 3], device=pixel_bundle.pls.device)
                if self.config.data.white_background else torch.zeros([1, 3], device=pixel_bundle.pls.device),
                is_training=False
            )
            loss = nn.functional.l1_loss(rendering_res.rgb, pixel_bundle.rgb_gt, reduction='sum') / \
                     (rendering_res.rgb.size(0) + 1e-5)
            print(f'register step: {i} loss: {loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def get_eval_dicts(self, img_pixel_bundle: RawPixelBundle, device: torch.device) -> \
            Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, np.ndarray]]:
        """
        Run evaluation on a single image
        :param device: trainer device
        :param img_pixel_bundle: pixel bundle of a single image, [H, W]
        :return: img_dict, metrics_dict, tensor_dict
        """

        if self.config.ray_generator.cam_opt_mode != "off" or self.config.ray_generator.pl_opt:
            if img_pixel_bundle.rgb_gt:
                self.register_view(img_pixel_bundle, device, steps=500)

        flat_pixel_bundle = img_pixel_bundle.flatten()
        has_gt = flat_pixel_bundle.rgb_gt is not None

        chunk = self.config.model.inference_chunk_size
        rets = []
        for i in range(0, flat_pixel_bundle.shape[0], chunk):
            ray_bundle = self.ray_generator.forward(flat_pixel_bundle[i:i + chunk].to(device))
            rendering_res = self.renderer.forward(
                ray_bundle,
                background_rgb=torch.ones([1, 3], device=device)
                if self.config.data.white_background else torch.zeros([1, 3], device=device),
                is_training=False
            )
            rets.append(rendering_res.to('cpu'))

        torch.set_default_tensor_type('torch.FloatTensor')
        ret = td_concat(rets)
        ret = ret.reshape(img_pixel_bundle.shape)
        rot = torch.linalg.inv(img_pixel_bundle.poses[0, 0, :3, :3])
        analytic_normals = torch.einsum('...ij,...i,...i->...j', ret.analytic_normals, ret.weights, ret.inside_sphere)
        analytic_normals = torch.matmul(rot[None, :], analytic_normals.reshape((-1, 3))[:, :, None]).reshape(
            ret.rgb.shape)
        normalized_analytic_normals = torch.einsum('...ij,...i,...i->...j', ret.normalized_analytic_normals,
                                                   ret.weights, ret.inside_sphere)
        normalized_analytic_normals = torch.matmul(rot[None, :],
                                                   normalized_analytic_normals.reshape((-1, 3))[:, :, None]).reshape(
            ret.rgb.shape)

        img_dict = {
            'rgb': ret.rgb.detach().cpu().numpy(),
            'analytic_normals': analytic_normals.detach().cpu().numpy(),
            'normalized_analytic_normals': normalized_analytic_normals.detach().cpu().numpy(),
        }
        if has_gt:
            img_dict['rgb_gt'] = img_pixel_bundle.rgb_gt.detach().cpu().numpy()
        if ret.visibilities is not None:
            img_dict['shadow_map'] = ret.visibilities.detach().cpu().numpy()
        metrics_dict = {
            'psnr': PSNR(ret.rgb, img_pixel_bundle.rgb_gt),
            'ssim': SSIM(ret.rgb, img_pixel_bundle.rgb_gt),
            'lpips': LPIPS(ret.rgb, img_pixel_bundle.rgb_gt),
        } if has_gt else {}
        tensor_dict = {
            'depth': ret.depth.detach().cpu().numpy(),
        }
        if ret.specular_cue is not None:
            tensor_dict['specular_hint'] = ret.specular_cue.detach().cpu().numpy()
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        return img_dict, metrics_dict, tensor_dict
