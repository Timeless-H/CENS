"""CENS model: Nerfacto plus rendering of only the pixels in a pattern (used by scripts/sim_script.py)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Type

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.models.base_model import Model
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig


@torch.no_grad()
def get_pattern_outputs_for_camera_ray_bundle(
    model: Model, camera_ray_bundle: RayBundle, pattern: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Render only the rays at flat indices ``pattern`` of an (H, W) camera ray bundle.

    Works with any nerfstudio model (including checkpoints trained as plain nerfacto).
    Returns (H, W, C) outputs with zeros at pixels not in ``pattern``.
    """
    input_device = camera_ray_bundle.directions.device
    num_rays_per_chunk = model.config.eval_num_rays_per_chunk
    image_height, image_width = camera_ray_bundle.origins.shape[:2]
    num_rays = pattern.numel()
    flat_bundle = camera_ray_bundle.flatten()
    outputs_lists = defaultdict(list)
    for start_idx in range(0, num_rays, num_rays_per_chunk):
        pattern_chunk = pattern.view(-1)[start_idx : start_idx + num_rays_per_chunk]
        ray_bundle = flat_bundle[pattern_chunk].to(model.device)
        outputs = model.forward(ray_bundle=ray_bundle)
        for output_name, output in outputs.items():  # type: ignore
            if not isinstance(output, torch.Tensor):
                continue
            outputs_lists[output_name].append(output.to(input_device))
    outputs = {}
    for output_name, outputs_list in outputs_lists.items():
        concat_output = torch.cat(outputs_list)
        rec = torch.zeros((image_height * image_width, concat_output.shape[-1]), device=input_device)
        rec[pattern] = concat_output
        outputs[output_name] = rec.view(image_height, image_width, -1)
    return outputs


@dataclass
class CENSModelConfig(NerfactoModelConfig):
    """CENS model config (Nerfacto hyperparameters unchanged)."""

    _target: Type = field(default_factory=lambda: CENSModel)
    """Target class to instantiate."""


class CENSModel(NerfactoModel):
    """Nerfacto with pattern-restricted rendering."""

    config: CENSModelConfig

    def get_pattern_outputs_for_camera_ray_bundle(
        self, camera_ray_bundle: RayBundle, pattern: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """See :func:`get_pattern_outputs_for_camera_ray_bundle`."""
        return get_pattern_outputs_for_camera_ray_bundle(self, camera_ray_bundle, pattern)
