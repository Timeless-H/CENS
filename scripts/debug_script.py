"""Launch a CENS training run from Python (equivalent CLI: ``ns-train cens ...``).

Set CENS_DEBUGPY=1 to wait for a debugger on port 5678.
"""

import math
import os
import random
from pathlib import Path

import numpy as np
import torch

from nerfstudio.configs.method_configs import all_methods
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.trainer import TrainerConfig

if os.environ.get("CENS_DEBUGPY") == "1":
    import debugpy

    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger...")
    debugpy.wait_for_client()


def reduce_config(config: TrainerConfig, tmp_path: Path) -> TrainerConfig:
    """Run-specific settings on top of the ``cens`` method defaults."""
    config.max_num_iterations = 234100  # 10% --> 93912; 25% --> 234780
    config.pipeline.datamanager.dataparser = NerfstudioDataParserConfig(data=Path("data/Jamtal_temporal/nerf-prep-25_06"))
    config.pipeline.datamanager.dataparser.downscale_factor = 2
    config.pipeline.datamanager.train_num_rays_per_batch = 4096
    config.pipeline.datamanager.eval_num_rays_per_batch = 4096

    ## canny & uniform grid sampling (uniform_grid_sampling=True is the cens default)
    config.pipeline.datamanager.uniform_grid_sampling = True
    config.pipeline.datamanager.mask_pattern = [4, 1]  # [3, 4, 1]
    # canny phase = first 30% of total iters
    config.pipeline.datamanager.user_defined_max_iter = math.floor(config.max_num_iterations * 0.3)
    if config.pipeline.datamanager.mask_pattern[0] == 3:
        config.max_num_iterations = config.max_num_iterations + config.pipeline.datamanager.user_defined_max_iter

    ## multiprocessing
    config.pipeline.datamanager.dataloader_num_workers = 4
    config.pipeline.datamanager.prefetch_factor = 2

    config.pipeline.model.camera_optimizer.mode = "off"  # also the cens default

    config.vis = "viewer+tensorboard"
    config.logging.relative_log_dir = Path("/tmp/")

    config.experiment_name = "jamtal_t2"
    config.set_timestamp()
    config.output_dir = tmp_path / "outputs"
    config.print_to_terminal()
    config.save_config()
    return config


def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_loop(local_rank: int, world_size: int, config: TrainerConfig):
    """Set up and run the trainer for one process."""
    _set_random_seed(config.machine.seed)
    trainer = config.setup(local_rank=local_rank, world_size=world_size)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    selected_config = reduce_config(all_methods["cens"], tmp_path=Path("/wkspace/"))
    train_loop(local_rank=0, world_size=1, config=selected_config)
    print("Debugging complete. Exiting...")
