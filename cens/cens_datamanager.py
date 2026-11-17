"""CENS datamanager: ParallelDataManager whose ray stream uses the CENS (UGS) pixel sampler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Type

from torch.utils.data import DataLoader

from nerfstudio.cameras.cameras import CameraType
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager, ParallelDataManagerConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import PixelSampler
from nerfstudio.data.utils.data_utils import identity_collate
from nerfstudio.data.utils.dataloaders import RayBatchStream, variable_res_collate
from nerfstudio.utils.rich_utils import CONSOLE

from cens.cens_pixel_sampler import CENSPixelSamplerConfig, worker_num_images


class CENSRayBatchStream(RayBatchStream):
    """RayBatchStream that builds a :class:`CENSPixelSampler` in each worker.

    ``train_step_value`` is written by the datamanager in the main process before every train batch.
    It is only visible to the sampler when ``num_workers == 0`` (see ``CENSPixelSampler.sample``).
    """

    def __init__(
        self,
        *args,
        pixel_sampler_config: Optional[CENSPixelSamplerConfig] = None,
        ugs: bool = False,
        mask_pattern: Optional[List[int]] = None,
        user_defined_max_iters: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pixel_sampler_config = pixel_sampler_config if pixel_sampler_config is not None else CENSPixelSamplerConfig()
        self.ugs = ugs
        self.mask_pattern = mask_pattern
        self.user_defined_max_iters = user_defined_max_iters
        self.train_step_value: int = 0

    def _get_pixel_sampler(self, dataset: InputDataset, num_rays_per_batch: int) -> PixelSampler:
        """Same as RayBatchStream._get_pixel_sampler plus the CENS kwargs. Runs inside each worker."""
        is_equirectangular = (dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value).all()
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")
        fisheye_crop_radius = None
        if dataset.cameras.metadata is not None:
            fisheye_crop_radius = dataset.cameras.metadata.get("fisheye_crop_radius")

        n_images = worker_num_images(dataset.cameras.shape[0])
        return self.pixel_sampler_config.setup(
            is_equirectangular=is_equirectangular,
            num_rays_per_batch=num_rays_per_batch,
            fisheye_crop_radius=fisheye_crop_radius,
            uniform_grid_sampling=self.ugs,
            mask_pattern=self.mask_pattern,
            user_defined_max_iters=self.user_defined_max_iters,
            img_nhw=(n_images, int(dataset.cameras.height[0, 0]), int(dataset.cameras.width[0, 0])),
            step_provider=self,
        )


@dataclass
class CENSDataManagerConfig(ParallelDataManagerConfig):
    """ParallelDataManager config with CENS uniform grid sampling options."""

    _target: Type = field(default_factory=lambda: CENSDataManager)
    """Target class to instantiate."""
    pixel_sampler: CENSPixelSamplerConfig = field(default_factory=CENSPixelSamplerConfig)
    """CENS pixel sampler configuration."""
    uniform_grid_sampling: bool = True
    """Whether to use uniform grid sampling (UGS) instead of random pixel sampling."""
    mask_pattern: List[int] = field(default_factory=lambda: [4, 1])
    """Indices into pixel_sampler.mask_pattern_name: [main, support] or [mask_canny(3), main, support]."""
    user_defined_max_iter: int = 3000
    """Length of the Canny phase in steps (only used if mask_pattern[0] == 3 / mask_canny)."""


class CENSDataManager(ParallelDataManager[InputDataset]):
    """ParallelDataManager feeding the train step to the CENS sampler."""

    config: CENSDataManagerConfig

    def setup_train(self):
        """Same as ParallelDataManager.setup_train, with a CENSRayBatchStream."""
        self.train_raybatchstream = CENSRayBatchStream(
            input_dataset=self.train_dataset,
            num_rays_per_batch=self.config.train_num_rays_per_batch,
            num_images_to_sample_from=(
                50
                if self.config.load_from_disk and self.config.train_num_images_to_sample_from == float("inf")
                else self.config.train_num_images_to_sample_from
            ),
            num_times_to_repeat_images=(
                10
                if self.config.load_from_disk and self.config.train_num_times_to_repeat_images == float("inf")
                else self.config.train_num_times_to_repeat_images
            ),
            device=self.device,
            collate_fn=variable_res_collate,
            load_from_disk=self.config.load_from_disk,
            custom_ray_processor=self.custom_ray_processor,
            pixel_sampler_config=self.config.pixel_sampler,
            ugs=self.config.uniform_grid_sampling,
            mask_pattern=self.config.mask_pattern,
            user_defined_max_iters=self.config.user_defined_max_iter,
        )
        self.train_ray_dataloader = DataLoader(
            self.train_raybatchstream,
            batch_size=1,
            num_workers=self.config.dataloader_num_workers,
            prefetch_factor=self.config.prefetch_factor,
            shuffle=False,
            collate_fn=identity_collate,
        )
        self.iter_train_raybundles = iter(self.train_ray_dataloader)

    def next_train(self, step: int):
        """Publish the train step to the stream (effective with num_workers == 0), then fetch the batch."""
        self.train_raybatchstream.train_step_value = self.train_count
        return super().next_train(step)
