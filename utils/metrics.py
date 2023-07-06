import warnings
import torch
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from lpips import LPIPS as _LPIPS


@torch.no_grad()
def PSNR(rgb, rgb_gt):
    return float(peak_signal_noise_ratio(rgb, rgb_gt, data_range=1.0).item())

@torch.no_grad()
def SSIM(rgb, rgb_gt):
    rgb = torch.moveaxis(rgb, -1, 0)[None, ...]
    rgb_gt = torch.moveaxis(rgb_gt, -1, 0)[None, ...]
    return float(structural_similarity_index_measure(rgb, rgb_gt, data_range=1.0).item())

@torch.no_grad()
def LPIPS(rgb, rgb_gt):
    rgb = torch.moveaxis(rgb, -1, 0)[None, ...]
    rgb_gt = torch.moveaxis(rgb_gt, -1, 0)[None, ...]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lpips = _LPIPS(net='alex', verbose=False).cpu()
    return float(lpips(rgb, rgb_gt, normalize=True).item())
