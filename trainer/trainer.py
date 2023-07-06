import itertools
import os
import pathlib
import pickle
import random
from dataclasses import asdict
from typing import Union, cast, Dict

import imageio.v3
import numpy as np
import torch
import torch.distributed
import trimesh
import tyro
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary
from tqdm import tqdm

from configs.main_config import SystemConfig
from data.data_loader import NRSHMDataManager, PixelSamplingStrategy
from data.shm_helper import NRDataSHMInfo
from pipelines.base_pipeline import BaseNRHintPipeline
from trainer.ddp_helper import local_rank


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # it just can work on non-cuda env...


class Trainer:
    _pipeline: Union[BaseNRHintPipeline, DDP]

    def __init__(self, config: SystemConfig, shm_info: NRDataSHMInfo):
        """
        init object, load data from shm_info,
        :param config: SystemConfig passed from launcher
        :param shm_info: shm_info passed from launcher, used to access data in shm
        """
        self.config = config

        # setup cuda & distributed
        self.rank = local_rank()
        self.world_size = 1  # default case
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            torch.backends.cudnn.benchmark = True  # type: ignore

            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                torch.distributed.init_process_group(backend='nccl', init_method='env://')
                self.world_size = torch.distributed.get_world_size()
                torch_rank = torch.distributed.get_rank()
                assert torch_rank == self.rank, \
                    f"torch_rank {torch_rank} != rank {self.rank}, initialization might have failed..."
                self.device = torch.device('cuda', self.rank)
                torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')

        # init logging
        self.log_dir = pathlib.Path(self.config.base_dir) / self.config.exp_name / self.config.scene_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.is_main_process:
            with open(self.log_dir / 'config.yaml', 'w') as f:
                f.write(tyro.to_yaml(self.config))
            wandb.init(
                project="NR2023",
                name=f'{config.exp_name}_{config.scene_name}',
                group=config.exp_name,
                notes="NR Hint Relighting",
                config=asdict(config),
                resume="allow",
                id=str(self.log_dir).replace("/", "_")
            )
            wandb.run.log_code()  # log all files in the current directory

        # Seed everything
        seed_everything(config.seed + self.rank)

        # Init pipeline
        self._pipeline = BaseNRHintPipeline(config, shm_info).to(self.device)
        if self.world_size > 1:
            self._pipeline = DDP(
                self._pipeline,
                device_ids=[self.rank],
                find_unused_parameters=False,  # to speed up && manually ensured non-redundant params in pipeline
                broadcast_buffers=True  # to ensure all processes have the same initial noise in buffers
            )

        # Log model summary
        if self.is_main_process:
            summary(self.pipeline)

        self.optimizer = torch.optim.Adam(self.pipeline.get_param_groups())

        alpha = config.model.lr_alpha
        warm_up_end = config.model.warm_up_end
        end_iter = config.model.end_iter

        def lr_lambda(iter_step):
            if iter_step < warm_up_end:
                learning_factor = iter_step / warm_up_end
            else:
                progress = (iter_step - warm_up_end) / (end_iter - warm_up_end)
                learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
            return learning_factor

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Setup data manager
        self.data_manager = NRSHMDataManager(
            shm_info=shm_info,
            batch_size=config.model.batch_size // self.world_size,
            strategy=PixelSamplingStrategy.ALL_IMAGES,
            image_idx_rng_seed=config.seed,  # local rank will be handled by sampler
            pixel_idx_rng_seed=config.seed,  # local rank will be handled by sampler
            local_rank=self.rank
        )

        # Load ckpts
        self.global_step = 0
        self.load_ckpt()

    @property
    def pipeline(self):
        """Returns the unwrapped model if in ddp"""
        if isinstance(self._pipeline, DDP):
            return cast(BaseNRHintPipeline, self._pipeline.module)
        return self._pipeline

    @property
    def is_main_process(self):
        return self.rank == 0

    @property
    def use_ddp(self):
        return self.world_size > 1

    def wait_all(self):
        if self.use_ddp:
            torch.distributed.barrier()

    @property
    def model_states(self) -> Dict:
        """Returns the model states to be saved in checkpoint"""

        return {
            "world_size": self.world_size,
            "global_step": self.global_step,
            "pipeline": self.pipeline.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict()
        }

    @property
    def rng_states(self):
        """Returns the rng states to be saved in checkpoint"""

        return {
            "python.random": random.getstate(),
            "np.random": np.random.get_state(),
            "torch.random": torch.random.get_rng_state(),
            "torch.cuda.random": torch.cuda.random.get_rng_state(self.device),
            "ray_generator.image": self.data_manager.sampler.image_rng.__getstate__(),
            "ray_generator.pixel": self.data_manager.sampler.pixel_rng.__getstate__()
        }

    def save_ckpt(self):
        """Save checkpoint"""

        if self.is_main_process:
            ckpt_path = self.log_dir / "ckpt" / f"step_{self.global_step:07d}.ckpt"
            torch.save(self.model_states, ckpt_path)
        # also save rng states
        rng_state_path = self.log_dir / "rng_state" / f"step_{self.global_step:07d}_device_{self.rank}.pickle"
        pickle.dump(self.rng_states, open(rng_state_path, 'wb'))

    # noinspection PyBroadException
    def load_ckpt(self):
        """Load checkpoint"""

        # compose ckpt paths
        ckpt_path = self.log_dir / "ckpt"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        rng_state_path = self.log_dir / "rng_state"
        rng_state_path.mkdir(parents=True, exist_ok=True)

        if self.config.ckpt_path is not None:
            ckpts = [self.config.ckpt_path]
            print(f'Loading given ckpt from {self.config.ckpt_path}')
        else:
            ckpts = [os.path.join(ckpt_path, f) for f in sorted(os.listdir(ckpt_path)) if 'ckpt' in f]
            print(f'Found {len(ckpts)} ckpts in {ckpt_path}')
        if len(ckpts) > 0:
            try:
                ckpt_path = ckpts[-1]
                print(f'Resume from ckpt: {ckpt_path}')
                last_world_size = self._load_ckpt_file(ckpt_path)
            except EOFError:  # in case last ckpt is corrupted
                ckpt_path = ckpts[-2]
                print(f'Retrying resume from ckpt: {ckpt_path}')
                last_world_size = self._load_ckpt_file(ckpt_path)

            if last_world_size == self.world_size:
                try:
                    rng_state_path = rng_state_path / f"step_{self.global_step:07d}_device_{self.rank}.pickle"
                    self._load_rng_states(rng_state_path)
                except Exception as e:
                    print(e)
                    print("rng state resume failed, the results might not be fully reproducible")

    def _load_ckpt_file(self, ckpt_file):
        """Load checkpoint from specific file"""

        ckpt = torch.load(ckpt_file, map_location=self.device)
        self.global_step = ckpt["global_step"]
        self.pipeline.load_state_dict(ckpt["pipeline"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt["scheduler"])
        return ckpt["world_size"]

    def _load_rng_states(self, rng_state_path):
        """Load rng states from specific file"""

        rng_states = pickle.load(open(rng_state_path, "rb"))
        random.setstate(rng_states["python.random"])
        np.random.set_state(rng_states["np.random"])
        torch.random.set_rng_state(rng_states["torch.random"])
        torch.cuda.set_rng_state(rng_states["torch.cuda.random"])
        self.data_manager.sampler.image_rng.__setstate__(rng_states["ray_generator.image"])
        self.data_manager.sampler.pixel_rng.__setstate__(rng_states["ray_generator.pixel"])

    def run(self):
        """Core entry point for training"""

        # training
        if not self.config.evaluation_only:
            start_step = self.global_step
            for _ in tqdm(
                    range(start_step, self.config.model.end_iter),
                    desc=f"Training: ",
                    initial=start_step,
                    total=self.config.model.end_iter,
                    dynamic_ncols=True,
                    disable=(self.rank != 0)
            ):
                loss_dict = self.train_iter()
                if self.global_step % self.config.intervals.log_metrics == 0 and self.is_main_process:
                    wandb.log(loss_dict, step=self.global_step)
                self.global_step += 1
                if self.global_step % self.config.intervals.save_ckpt == 0:
                    self.save_ckpt()
                if self.global_step % self.config.intervals.render_test_views == 0:
                    self.render_test_views()
                if self.global_step % self.config.intervals.dump_mesh == 0:
                    self.dump_mesh()
                if self.global_step % self.config.intervals.render_video == 0:
                    self.render_video()

        # final dumps / evaluation
        self.dump_mesh(resolution=1024)
        self.render_test_views(is_final=True)

    def train_iter(self):
        """One training iteration"""

        pixel_bundle = self.data_manager.next_train_batch()
        pixel_bundle = pixel_bundle.to(self.device)
        rendering_res = self._pipeline.forward(pixel_bundle, global_step=self.global_step)  # use the warped model
        loss_dict = self.pipeline.get_train_loss_dict(rendering_res, pixel_bundle)
        loss = loss_dict['loss']

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss_dict

    def render_test_views(self, is_final=False):
        """Eval on selected images, save all into disk, and log one of them onto wandb"""

        total_test_view = self.data_manager.test_view_num()
        skip_num = self.config.data.testset_skip if not is_final else 1
        start_idx = self.rank * skip_num

        metrics_dicts = []
        for idx in tqdm(range(start_idx, total_test_view, skip_num * self.world_size),
                        desc=f"Rendering test views on process {self.rank}: "):
            metrics_dict = self.render_single_view(idx)
            metrics_dicts.append(metrics_dict)

        self.wait_all()

        # gather metrics
        if self.use_ddp:
            output_list = [None for _ in range(self.world_size)]
            torch.distributed.gather_object(
                obj=metrics_dicts,
                object_gather_list=output_list if self.is_main_process else None,
                dst=0
            )
        else:
            output_list = [metrics_dicts]

        # calculate mean metrics & log onto wandb
        if self.is_main_process:
            gather_output = {}
            image_cnt = 0
            for item in itertools.chain(*output_list):  # type: ignore
                image_cnt += 1
                for k, v in item.items():
                    gather_output.setdefault(k, 0.)
                    gather_output[k] += v
            final_output = {}
            for k, v in gather_output.items():
                final_output[f'val/{k}' if is_final else f'val/{k}'] = v / image_cnt
            wandb.log(final_output, step=self.global_step)

        self.wait_all()

    @torch.no_grad()
    def render_single_view(self, view_index, is_training_view: bool = False):
        """Render single view"""

        img_pixel_bundle = self.data_manager.get_test_view(view_index)
        # only eval on single gpu
        img_dict, metrics_dict, tensor_dict = self.pipeline.get_eval_dicts(img_pixel_bundle, self.device)
        self.save_dumps(view_index, img_dict, tensor_dict)
        if view_index == 0 and self.is_main_process:
            self.log_images(img_dict)
        return metrics_dict

    def save_dumps(self, view_idx, image_dict, tensor_dict):
        """Save dumped images and tensors to disk"""

        dump_dir = self.log_dir / "test_views" / f"step_{self.global_step:07d}"
        dump_dir.mkdir(parents=True, exist_ok=True)

        # dump images
        for k, v in image_dict.items():
            if "normal" in k:
                v = v * 0.5 + 0.5  # scale normal maps from [-1, 1] to [0, 1]
            if v.shape[-1] == 1:
                v = v[..., 0]
            imageio.v3.imwrite(dump_dir / f"{k}_{view_idx:03d}.png", (v * 255).clip(0, 255).astype(np.uint8))

        # dump tensors into npy
        for k, v in tensor_dict.items():
            np.save(dump_dir / f"{k}_{view_idx:03d}.npy", v)  # type: ignore

    def log_images(self, image_dict):
        """Log images to wandb"""

        for k, v in image_dict.items():
            if "normal" in k:
                v = v * 0.5 + 0.5  # scale normal maps from [-1, 1] to [0, 1]
            wandb.log({k: wandb.Image((v * 255).clip(0, 255).astype(np.uint8))}, step=self.global_step)

    def dump_mesh(self, resolution: int = 256):
        """Dump mesh to disk"""

        if self.is_main_process:
            mesh_dir = self.log_dir / "mesh"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            mesh_file_path = mesh_dir / f'step_{self.global_step:07d}_res_{resolution}.obj'
            bound_min = torch.tensor([-1.01, -1.01, -1.01], dtype=torch.float32)
            bound_max = torch.tensor([1.01, 1.01, 1.01], dtype=torch.float32)
            vertices, triangles = self.pipeline.renderer. \
                extract_geometry(bound_min, bound_max, resolution=resolution, threshold=0.)
            mesh = trimesh.Trimesh(vertices, triangles)
            mesh.export(mesh_file_path)
            # wandb.log({'mesh': wandb.Object3D(str(mesh_file_path))}, step=self.global_step)
            # will cost too much space on wandb cloud, uncomment if you are rich
        self.wait_all()

    def render_video(self):
        """Render video"""

        video_dir = self.log_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_frame_dir = video_dir / f'step_{self.global_step:07d}'
        video_frame_dir.mkdir(parents=True, exist_ok=True)
        video_pixel_bundle = self.data_manager.get_video_pixel_bundle(self.config.data.video_frame_num, is_z_up=self.config.data.is_z_up)

        if self.is_main_process:
            video_frame_buffer = torch.empty(*video_pixel_bundle.shape, 3, dtype=torch.uint8, device='cpu')

        assert len(video_pixel_bundle) % self.world_size == 0, "video frame number should be divisible by world size"
        frames_per_process = len(video_pixel_bundle) // self.world_size
        for i in tqdm(range(frames_per_process), desc=f"Rendering video frames on process {self.rank}: "):
            idx = i + frames_per_process * self.rank
            img_dict, _, _ = self.pipeline.get_eval_dicts(video_pixel_bundle[idx], self.device)
            uint8_rgb = (img_dict['rgb'] * 255).clip(0, 255).astype(np.uint8)
            imageio.v3.imwrite(video_frame_dir / f'{idx:03d}.png', uint8_rgb)
            if self.is_main_process:
                video_frame_buffer[idx] = torch.tensor(uint8_rgb, dtype=torch.uint8, device='cpu')
                single_idx_buffer = torch.empty((1,), dtype=torch.long, device=self.device)
                single_frame_buffer = torch.empty_like(video_frame_buffer[idx], dtype=torch.uint8, device=self.device)
                for sub_process_rank in range(1, self.world_size):
                    torch.distributed.recv(tensor=single_idx_buffer, src=sub_process_rank, tag=0)
                    torch.distributed.recv(tensor=single_frame_buffer, src=sub_process_rank, tag=1)
                    video_frame_buffer[single_idx_buffer.item()] = single_frame_buffer.cpu()
            else:
                torch.distributed.send(torch.tensor([idx], device=self.device), dst=0, tag=0)
                torch.distributed.send(torch.tensor(uint8_rgb, dtype=torch.uint8, device=self.device), dst=0, tag=1)
            self.wait_all()

        if self.is_main_process:
            video_rgb = video_frame_buffer.numpy().astype(np.uint8)
            imageio.v3.imwrite(os.path.join(video_dir, f'step_{self.global_step:06d}_rot_view.mp4'),
                               (video_rgb[:self.config.data.video_frame_num]), fps=30, quality=9)
            imageio.v3.imwrite(os.path.join(video_dir, f'step_{self.global_step:06d}_rot_light.mp4'),
                               (video_rgb[self.config.data.video_frame_num:]), fps=30, quality=9)

        self.wait_all()

    def release_shm(self):
        self.data_manager.release_shm()
        del self.data_manager
