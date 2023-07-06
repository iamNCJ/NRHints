from dataclasses import dataclass, field
from typing import Optional, Union
import tyro

from camera.ray_generator import RayGeneratorConfig
from data.data_config import DataManagerConfig
from models.neus_hint_model import NeuSModelConfig, NeuSRendererConfig


@dataclass(frozen=True)
class IntervalsConfig:
    """Configuration for intervals, like saving checkpoints and testing."""

    log_metrics: int = 200
    """Log metrics every N iterations."""
    save_ckpt: int = 5_000
    """Save checkpoint every N iterations."""
    render_test_views: int = 250_000
    """Test every N iterations."""
    render_video: int = 1_000_000
    """Render video every N iterations."""
    dump_mesh: int = 500_000
    """Dump mesh every N iterations."""


@dataclass(frozen=True)
class SystemConfig:
    """Configuration for the whole system."""

    model: NeuSModelConfig = NeuSModelConfig()
    data: DataManagerConfig = DataManagerConfig()
    ray_generator: RayGeneratorConfig = RayGeneratorConfig()
    intervals: IntervalsConfig = IntervalsConfig()

    ckpt_path: Optional[str] = None
    """Path to the checkpoint to load."""
    base_dir: str = 'outputs'
    """Base directory for saving outputs."""
    exp_name: str = 'baseline'
    """Name of the experiment setup."""
    scene_name: str = 'scene'
    """Name of the scene, can be used for other remarks."""

    seed: int = 3407
    """Random seed for reproducibility."""
    serialized_shm_info: Optional[str] = None
    """Serialized SHM info as srt arg."""
    evaluation_only: bool = False
    """Whether to only run evaluation."""


@dataclass(frozen=True)
class NRHints(SystemConfig):
    """NeuS Relighting with Hints"""

    pass


@dataclass(frozen=True)
class NRHintsCamOpt(SystemConfig):
    """NeuS Relighting with Hints and Camera Optimization"""

    ray_generator: RayGeneratorConfig = field(default=RayGeneratorConfig(cam_opt_mode="SO3xR3"))


@dataclass(frozen=True)
class PLNaive(SystemConfig):
    """NeuS Relighting with Naive Point Light"""

    model: NeuSModelConfig = field(default=NeuSModelConfig(
        renderer=NeuSRendererConfig(
            shadow_hint=False,
            specular_hint=False
        )
    ))


@dataclass
class MainArgs:
    # Turn off flag conversion for booleans with default values.
    config: tyro.conf.FlagConversionOff[Union[NRHints, PLNaive, NRHintsCamOpt]]


def get_config():
    return tyro.cli(MainArgs, default=MainArgs(config=NRHintsCamOpt()))


if __name__ == '__main__':
    config = get_config()
    print(config)
