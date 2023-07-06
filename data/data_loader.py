from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import torch
from jaxtyping import Float, Int

from camera.video_pose_utils import gen_fix_light_rot_view, gen_fix_view_rot_light
from data.shm_helper import NRDataSHMArrayReader, NRDataSHMInfo
from utils.tensor_dataclass import TensorDataclass


class PixelSamplingStrategy(Enum):
    """
    Strategy for sampling batch of pixels from images.
    """

    ALL_IMAGES = 'all_images'
    """Pixels in a batch are sampled from different images, like NeRF."""
    SAME_IMAGE = 'same_image'
    """Pixels in a batch are sampled from the same image, like NeuS."""


class PixelSampler(object):
    """
    Sample rays (pixels) from images.
    """

    def __init__(
            self,
            shm_info: NRDataSHMInfo,
            batch_size: int,
            image_idx_rng_seed: int = 42,
            pixel_idx_rng_seed: int = 42,
            local_rank: int = 0,
            strategy: PixelSamplingStrategy = PixelSamplingStrategy.SAME_IMAGE,
            training_view_num_limit: Optional[int] = None,
    ):
        """
        Initialize ray sampler.
        :param shm_info:
        :param batch_size:
        :param image_idx_rng_seed: set same seed on each process so that they sample the same image
        :param pixel_idx_rng_seed: process rank will be added to ensure each process sample a different batch
        :param local_rank: process local rank, to ensure each process sample a different batch
        :param strategy: RaySamplingStrategy
        """

        self.batch_size = batch_size
        self.train_image_num = shm_info.num_image_per_split[0] if training_view_num_limit is None \
            else training_view_num_limit
        self.H, self.W = shm_info.camera.H, shm_info.camera.W
        self.total_image_num = shm_info.total_image_num
        self.strategy = strategy
        if strategy == PixelSamplingStrategy.ALL_IMAGES:
            image_idx_rng_seed += local_rank
            # when using SINGLE_IMAGE strategy, all processes should use the same image each batch
            # hence will not add rank bias to seed for SINGLE_IMAGE strategy
        pixel_idx_rng_seed += local_rank
        self.image_rng = np.random.default_rng(seed=image_idx_rng_seed)
        self.pixel_rng = np.random.default_rng(seed=pixel_idx_rng_seed)
        print(f'Local rank: {local_rank}, image_rng_seed: {image_idx_rng_seed}, pixel_rng_seed: {pixel_idx_rng_seed}')

    def sample_batch(self):
        # TODO: sample more inside mask
        if self.strategy == PixelSamplingStrategy.ALL_IMAGES:
            img_indices = self.image_rng.choice(self.train_image_num, self.batch_size)
        elif self.strategy == PixelSamplingStrategy.SAME_IMAGE:
            img_indices = self.image_rng.choice(self.train_image_num, 1)
            img_indices = np.repeat(img_indices, self.batch_size)
        else:
            raise NotImplementedError
        h_indices = self.pixel_rng.choice(self.H, self.batch_size)
        w_indices = self.pixel_rng.choice(self.W, self.batch_size)

        return img_indices, h_indices, w_indices


@dataclass
class RawPixelBundle(TensorDataclass):
    img_indices: Optional[Int[torch.Tensor, "*bs 1"]]
    h_indices: Float[torch.Tensor, "*bs 1"]
    w_indices: Float[torch.Tensor, "*bs 1"]
    poses: Float[torch.Tensor, "*bs 4 4"]
    pls: Float[torch.Tensor, "*bs 3"]
    rgb_gt: Optional[Float[torch.Tensor, "*bs 3"]]

    _field_custom_dimensions = {"poses": 2}


class VideoPixelBundle(object):
    """Generate pixel bundles for video frames."""

    def __init__(self, video_poses, video_pls, H, W):
        """init bundle infos"""

        self.video_poses = video_poses
        self.video_pls = video_pls
        self.H, self.W = H, W

    def __len__(self):
        return len(self.video_poses)

    @property
    def shape(self):
        return len(self), self.H, self.W

    def __getitem__(self, idx: int):
        """return assembled pixel bundle for a frame"""

        H, W = self.H, self.W
        w_indices, h_indices = torch.meshgrid(
            torch.linspace(0, W - 1, W),
            torch.linspace(0, H - 1, H),
            indexing='xy')

        return RawPixelBundle(
            img_indices=None,
            h_indices=h_indices[..., None],
            w_indices=w_indices[..., None],
            rgb_gt=None,
            poses=torch.from_numpy(self.video_poses[idx])[None, None].repeat(H, W, 1, 1),
            pls=torch.from_numpy(self.video_pls[idx])[None, None].repeat(H, W, 1)
        )


class NRSHMDataManager(object):
    def __init__(
            self,
            shm_info: NRDataSHMInfo,
            batch_size: int,
            strategy: PixelSamplingStrategy = PixelSamplingStrategy.SAME_IMAGE,
            training_view_num_limit: Optional[int] = None,
            image_idx_rng_seed: int = 42,
            pixel_idx_rng_seed: int = 42,
            local_rank: int = 0
    ):
        self.stream = None
        self.sampler = PixelSampler(
            shm_info,
            batch_size,
            strategy=strategy,
            training_view_num_limit=training_view_num_limit,
            image_idx_rng_seed=image_idx_rng_seed,
            pixel_idx_rng_seed=pixel_idx_rng_seed,
            local_rank=local_rank
        )
        self.info = shm_info
        self.reader = NRDataSHMArrayReader(shm_info)
        self.imgs, self.poses, self.pls = self.reader.get_shm_arrays()

    def get_video_pixel_bundle(self, frame_num: int, is_z_up: bool = False) -> VideoPixelBundle:
        """generate video poses, return a mocked pixel bundle"""

        pls_dist = np.linalg.norm(self.pls, axis=-1, keepdims=True)
        pls_avg_dist = np.mean(pls_dist)
        eyes = self.poses[..., :3, -1]
        eye_dist = np.linalg.norm(eyes, axis=-1, keepdims=True)
        eye_avg_dist = np.mean(eye_dist)
        render_poses_0, render_pls_0 = gen_fix_light_rot_view(
            frame_num,
            eye_avg_dist,
            [0, 0.5 * pls_avg_dist, 0.866 * pls_avg_dist],
            is_z_up=is_z_up
        )
        render_pls_0 = render_pls_0[..., :3]  # only uses pl position
        render_poses_1, render_pls_1 = gen_fix_view_rot_light(
            frame_num,
            pls_avg_dist,
            [25, 25, 25],
            -180, -30,
            view_radius=eye_avg_dist,
            is_z_up=is_z_up
        )
        render_pls_1 = render_pls_1[..., :3]  # only uses pl position
        video_poses = np.concatenate([render_poses_0, render_poses_1], axis=0)
        video_pls = np.concatenate([render_pls_0, render_pls_1], axis=0)
        return VideoPixelBundle(video_poses, video_pls, self.info.camera.H, self.info.camera.W)

    def next_train_batch(self):
        """return a batch of sampled pixels."""

        img_indices, h_indices, w_indices = self.sampler.sample_batch()
        return RawPixelBundle(
            img_indices=torch.from_numpy(img_indices)[..., None],
            h_indices=torch.from_numpy(h_indices)[..., None],
            w_indices=torch.from_numpy(w_indices)[..., None],
            rgb_gt=torch.from_numpy(self.imgs[img_indices, h_indices, w_indices]),
            poses=torch.from_numpy(self.poses[img_indices]),
            pls=torch.from_numpy(self.pls[img_indices])
        )

    def test_view_num(self):
        """return the number of test views."""

        return self.info.num_image_per_split[2]

    def get_test_view(self, idx):
        """return a test view according to index."""

        idx += self.info.num_image_per_split[0] + self.info.num_image_per_split[1]
        return self[idx]

    def __getitem__(self, idx):
        H, W = self.info.camera.H, self.info.camera.W
        w_indices, h_indices = torch.meshgrid(
            torch.linspace(0, W - 1, W),
            torch.linspace(0, H - 1, H),
            indexing='xy')
        return RawPixelBundle(
            img_indices=torch.ones([H, W, 1], dtype=torch.long) * idx,
            h_indices=h_indices[..., None],
            w_indices=w_indices[..., None],
            rgb_gt=torch.from_numpy(self.imgs[idx]),
            poses=torch.from_numpy(self.poses[idx]).expand((H, W, 4, 4)),
            pls=torch.from_numpy(self.pls[idx]).expand((H, W, 3)),
        )

    def release_shm(self):
        """release the shared memory."""

        del self.imgs
        del self.poses
        del self.pls
        self.reader.release_shm()
        del self.reader


if __name__ == '__main__':
    from data.data_parser import parse_load_nr_data

    shm_data_writer = parse_load_nr_data('../../../datasets/Basket_Plane_PL_500')
    shm_info = shm_data_writer.get_shm_info()

    data_manager = NRSHMDataManager(shm_info, batch_size=512)
    for i in range(10):
        print(i)
        pixel_bundle = data_manager.next_train_batch()
        print(pixel_bundle.shape)
        print(pixel_bundle)

    for i in range(10):
        print(i)
        pixel_bundle = data_manager.get_test_view(i)
        print(pixel_bundle.shape)

    video_pixel_generator = data_manager.get_video_pixel_bundle(60)
    print(len(video_pixel_generator))
    print(video_pixel_generator.shape)
    print(video_pixel_generator[0].shape)

    data_manager.release_shm()
    shm_data_writer.release_shm()
