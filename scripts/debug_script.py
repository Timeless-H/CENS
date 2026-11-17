import debugpy 

# debugpy.listen(("", 0))  # You can change the port if needed
# print("Waiting for debugger...")
# debugpy.wait_for_client()  # Execution will pause here until a debugger connects

from pathlib import Path
import random
import math
import numpy as np
import torch

from nerfstudio.configs.method_configs import method_configs
# from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.vanilla_nerf import VanillaModelConfig
from nerfstudio.scripts.train import train_loop

# from add_mask_paths import validate_and_setup_masks

def reduce_config(config: TrainerConfig, tmp_path: Path):
    """
    Reduces the config to only the necessary parts for debugging.
    """
    config.max_num_iterations = 234100 # 11232 # reduced from 1000000 or 30000 ::10% --> 93912; 25% --> 234780
    ## reduce dataset params
    # config.pipeline.datamanager.dataparser = BlenderDataParserConfig(data=Path("data/synthetic_data/drums"))
    config.pipeline.datamanager.dataparser = NerfstudioDataParserConfig(data=Path("data/Jamtal_temporal/nerf-prep-25_06")) # "data/Frederik"
    config.pipeline.datamanager.dataparser.downscale_factor = 2
    config.pipeline.datamanager.train_num_rays_per_batch = 4096 # reduced from 1024
    config.pipeline.datamanager.eval_num_rays_per_batch = 4096 # reduced from 1024

    ## canny & uniform_grid sampling params
    config.pipeline.datamanager.uniform_grid_sampling = True
    config.pipeline.datamanager.mask_pattern = [4, 1] # [3, 4, 1] # mask patterns being used for sampling

    config.pipeline.datamanager.user_defined_max_iter = math.floor(config.max_num_iterations*0.3) # set canny iter limit to be first 30% of total iters
    if config.pipeline.datamanager.mask_pattern[0] == 3:
        config.max_num_iterations = config.max_num_iterations + config.pipeline.datamanager.user_defined_max_iter # add extra iterations for the canny pattern
    
    ## multiprocessing params
    config.pipeline.datamanager.dataloader_num_workers = 0 # reduced from 4
    config.pipeline.datamanager.prefetch_factor = None # reduced from 10 # None disables prefetching

    config.pipeline.model.camera_optimizer.mode = "off" # turn off camera optimization

    # use tensorboard for logging instead of wandb
    config.vis = "viewer+tensorboard"
    config.logging.relative_log_dir = Path("/tmp/")

    ## reduce model params
    if hasattr(config.pipeline.model, "num_coarse_samples"):
        assert isinstance(config.pipeline.model, VanillaModelConfig)
        config.pipeline.model.num_coarse_samples = 64 # reduced from 64

    if hasattr(config.pipeline.model, "num_importance_samples"):
        assert isinstance(config.pipeline.model, VanillaModelConfig)
        config.pipeline.model.num_importance_samples = 128 # reduced from 128

    # remove viewer
    # config.viewer.quit_on_train_completion = True

    # timestamp & output dir
    config.experiment_name = "jamtal_t2"
    config.set_timestamp()
    config.output_dir = tmp_path/"outputs"
    config.print_to_terminal()
    config.save_config()

    return config

def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_loop(local_rank: int, world_size: int, config: TrainerConfig):
    """
    Main training function that sets up and runs the trainer per process.
    Args:
        local_rank: Current rank of the process.
        world_size: Total number of gpus available.
        config: Configuration file specifying the training regimen for the trainer.
    """
    _set_random_seed(config.machine.seed)
    # Set up the trainer
    trainer = config.setup(local_rank=local_rank, world_size=world_size)
    trainer.setup()
    trainer.train()

if __name__ == "__main__":
    all_config_names = method_configs.keys()

    # selected_config = method_configs["vanilla-nerf"] 
    selected_config = method_configs["nerfacto"]

    tpath = Path("/wkspace/")
    selected_config = reduce_config(selected_config, tmp_path=tpath)

    # # Check and setup masks if mask_sobel (index 3) is enabled
    # status = validate_and_setup_masks(selected_config.pipeline.datamanager.mask_pattern, "/wkspace/data/Frederik")

    # if status['errors']:
    #     print("Setup failed:", status['errors'])
    # else:
    #     print("Setup successful, proceeding to train_loop...")
    train_loop(local_rank=0, world_size=1, config=selected_config)

    print("Debugging complete. Exiting...")