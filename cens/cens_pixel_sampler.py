"""CENS pixel sampler: uniform grid sampling (UGS) with optional Canny warm-up phase.

Schedule (``s`` = train step, ``M0`` = ``max_iters_in_mask0``, ``C`` = ``user_defined_max_iters``):

* pattern[0] != canny:   s < M0 -> pattern[0];   s >= M0 -> pattern[-1]
* pattern[0] == canny:   s < C  -> per-image canny masks;   C <= s < C+M0 -> pattern[1];   else -> pattern[-1]

Within a phase, step ``a`` (phase-local) takes slice ``[a*n, (a+1)*n)`` of a freshly shuffled pattern
(``n`` = rays per image). When the pattern is exhausted, a random "support" draw from pattern[-1] is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import tyro
from jaxtyping import Int
from torch import Tensor
from torch.utils.data import get_worker_info

from nerfstudio.data.pixel_samplers import PixelSampler, PixelSamplerConfig
from nerfstudio.data.utils.pixel_sampling_utils import divide_rays_per_image

from cens.cens_mask_patterns import flat_index_to_2d_coords, gen_active_indices_for_pattern, get_flat_active_indices

_GENERATORS: Dict[Tuple[str, int], torch.Generator] = {}


def get_ugs_generator(seed: int, device: str = "cuda") -> torch.Generator:
    """Per-process UGS generator, created lazily.

    It must never live on a config/dataset object: ``torch.Generator`` can't be unpickled in a
    spawned DataLoader worker (this was the multi-worker crash). One generator per (device, seed) per
    process, shared by all samplers in that process, as the original module-level generator was.
    """
    key = (device, seed)
    if key not in _GENERATORS:
        _GENERATORS[key] = torch.Generator(device=torch.device(device)).manual_seed(seed)
    return _GENERATORS[key]


@dataclass
class CENSPixelSamplerConfig(PixelSamplerConfig):
    """Configuration for :class:`CENSPixelSampler`."""

    _target: Type = field(default_factory=lambda: CENSPixelSampler)
    """Target class to instantiate."""
    ignore_mask: bool = True
    """Whether to ignore the masks when sampling (CENS uses masks only as Canny maps)."""
    mask_pattern_name: List[str] = field(
        default_factory=lambda: ["mask_101", "mask_010", "mask_all", "mask_canny", "mask_1000", "mask_0010"]
    )
    """Pattern names; ``mask_pattern`` indexes into this list."""
    rng_seed: int = 92
    """Seed of the per-process UGS generator."""
    rng_device: str = "cuda"
    """Device of the UGS generator (and of pattern generation). Changing it changes the sampled pixels."""
    # The following are set by CENSDataManagerConfig (passed as kwargs), hence hidden from the CLI.
    uniform_grid_sampling: tyro.conf.Suppress[bool] = False
    """Whether to use uniform grid sampling instead of random sampling."""
    mask_pattern: tyro.conf.Suppress[List[int]] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    """Indices into ``mask_pattern_name``: [main, (second,) support]."""
    user_defined_max_iters: tyro.conf.Suppress[int] = 3000
    """Length of the Canny phase in steps (only used if pattern[0] is mask_canny)."""


class CENSPixelSampler(PixelSampler):
    """Pixel sampler implementing CENS uniform grid sampling.

    Extra kwargs: ``uniform_grid_sampling``, ``mask_pattern``, ``user_defined_max_iters``,
    ``img_nhw`` (num images, H, W) and ``step_provider`` (object exposing ``train_step_value``).
    """

    config: CENSPixelSamplerConfig

    def __init__(self, config: CENSPixelSamplerConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        for key in ("user_defined_max_iters", "uniform_grid_sampling", "mask_pattern"):
            if self.kwargs.get(key) is not None:
                setattr(self.config, key, self.kwargs[key])
        self.step_provider = self.kwargs.get("step_provider")
        self._local_step = 0
        self._current_step = 0
        self.g = get_ugs_generator(self.config.rng_seed, self.config.rng_device)
        self.set_max_mask0_iterations(self.kwargs.get("img_nhw", (0, 0, 0)))

    def _name(self, pos: int) -> str:
        return self.config.mask_pattern_name[self.config.mask_pattern[pos]]

    def set_max_mask0_iterations(self, img_nhw: Tuple[int, int, int]) -> None:
        """Number of steps until the main pattern is exhausted for every image."""
        n, h, w = (int(v) for v in img_nhw)
        main_pattern = self._name(1) if self._name(0) == "mask_canny" else self._name(0)
        if main_pattern == "mask_all":
            self.max_iters_in_mask0 = n * (h * w // self.num_rays_per_batch)
        else:
            ugs_mask = gen_active_indices_for_pattern(h, w, main_pattern, self.g)
            self.max_iters_in_mask0 = n * (len(ugs_mask) // self.num_rays_per_batch)

    def sample_method_uniform(
        self,
        batch_size: int,
        img_idx: int,
        image_height: int,
        image_width: int,
        active_set_cnt: int,
        ugs_mask: Optional[Tensor] = None,
        support_mask: Optional[Tensor] = None,
        canny_img: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        """Take the ``active_set_cnt``-th slice of the pattern (or Canny map); fall back to support indices."""
        if canny_img is not None:
            # Fix: the generator lives on rng_device; masks from the dataset are on CPU (this used to crash).
            canny_indices = get_flat_active_indices(canny_img.to(self.g.device), image_width, generator=self.g)
            curr_active_indices = canny_indices[batch_size * active_set_cnt : batch_size * (active_set_cnt + 1)]
        else:
            assert ugs_mask is not None
            curr_active_indices = ugs_mask[batch_size * active_set_cnt : batch_size * (active_set_cnt + 1)]

        if len(curr_active_indices) < batch_size:
            if canny_img is not None and support_mask is None:
                support_mask = torch.randint(
                    low=0, high=len(canny_indices), size=(batch_size + 5,), device=canny_indices.device, generator=None
                )
            assert support_mask is not None
            curr_active_indices = support_mask[:batch_size]

        indices = flat_index_to_2d_coords(curr_active_indices, image_width, image_height, img_idx)
        return indices.to(device)

    def active_indices_support(self, image_width: int, image_height: int, batch_size: int) -> Tensor:
        """Random draw (with replacement) of ``batch_size + 5`` indices from the support pattern."""
        support_mask = gen_active_indices_for_pattern(image_height, image_width, self._name(-1), self.g)
        sel = torch.randint(low=0, high=len(support_mask), size=(batch_size + 5,), device=support_mask.device)
        return support_mask[sel]

    def _ugs_phase(self, step: int, h: int, w: int, max_rays_per_image: int, support_batch_size: int):
        """Returns (phase-local counter, shuffled pattern indices or None, support indices or None, canny phase)."""
        if self._name(0) == "mask_canny":
            udmi = self.config.user_defined_max_iters
            canny_grid_range = self.max_iters_in_mask0 + udmi
            if step < udmi:
                return step, None, None, True
            if step < canny_grid_range:
                as_cnt, pattern = step - udmi, self._name(1)
            else:
                as_cnt, pattern = step - canny_grid_range, self._name(-1)
        else:
            if step < self.max_iters_in_mask0:
                as_cnt, pattern = step, self._name(0)
            else:
                as_cnt, pattern = step - self.max_iters_in_mask0, self._name(-1)
        ugs_mask = gen_active_indices_for_pattern(h, w, pattern, self.g)
        support = None
        if (len(ugs_mask) // max_rays_per_image) <= as_cnt:
            support = self.active_indices_support(w, h, batch_size=support_batch_size)
        return as_cnt, ugs_mask, support, False

    def collate_image_dataset_batch_list(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        if not self.config.uniform_grid_sampling:
            return super().collate_image_dataset_batch_list(batch, num_rays_per_batch, keep_full_image)

        device = batch["image"][0].device
        num_images = len(batch["image"])
        assert num_rays_per_batch % 2 == 0, "num_rays_per_batch must be divisible by 2"
        num_rays_per_image = divide_rays_per_image(num_rays_per_batch, num_images)

        image_height, image_width, _ = batch["image"][0].shape
        as_cnt, ugs_mask, support, canny_phase = self._ugs_phase(
            self._current_step, image_height, image_width, max(num_rays_per_image), num_rays_per_image[0]
        )

        all_indices = []
        all_images: Dict[str, list] = {}
        for i, num_rays in enumerate(num_rays_per_image):
            image_height, image_width, _ = batch["image"][i].shape
            if self.config.is_equirectangular:
                indices = self.sample_method_equirectangular(num_rays, 1, image_height, image_width, device=device)
                indices[:, 0] = i
            else:
                canny_img = batch["mask"][i] if (canny_phase and "mask" in batch) else None
                indices = self.sample_method_uniform(
                    num_rays, i, image_height, image_width, as_cnt, ugs_mask, support, canny_img, device=device
                )
            all_indices.append(indices)
            for key, value in batch.items():
                if key in ["image_idx", "mask"]:
                    continue
                all_images.setdefault(key, []).append(value[i][indices[:, 1], indices[:, 2]])

        indices = torch.cat(all_indices, dim=0)
        collated_batch = {key: torch.cat(vals, dim=0) for key, vals in all_images.items()}
        assert collated_batch["image"].shape[0] == num_rays_per_batch

        c = indices[..., 0].flatten()
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices
        if keep_full_image:
            collated_batch["full_image"] = batch["image"]
        return collated_batch

    def sample(self, image_batch: Dict):
        """Sample a pixel batch; the UGS step comes from the step provider (main process) or a local counter.

        * ``num_workers == 0``: the datamanager writes the train step into the stream before each batch
          (eval batches reuse it), identical to the original implementation.
        * In a DataLoader worker the main process can't reach this object, so each worker counts the
          batches it produced; with ``max_iters_in_mask0`` computed on the worker's image slice this
          gives the same phase switch points in global steps.
        """
        if self.step_provider is not None and get_worker_info() is None:
            self._current_step = int(self.step_provider.train_step_value)
        else:
            self._current_step = self._local_step
        self._local_step += 1
        return super().sample(image_batch)


def worker_num_images(num_images: int) -> int:
    """Number of images in this DataLoader worker's partition (mirrors RayBatchStream.__iter__)."""
    info = get_worker_info()
    if info is None or num_images < info.num_workers:
        return num_images
    per_worker = int(math.ceil(num_images / float(info.num_workers)))
    return len(range(num_images)[info.id * per_worker : info.id * per_worker + per_worker])
