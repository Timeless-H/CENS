"""``ns-export-cens``: point cloud export for cens runs (stock ``ns-export`` is left untouched).

Differences to stock ``ns-export pointcloud``:
* the export rays can be sampled with a chosen UGS pattern (``--mask-pattern``),
* the world-frame conversion copies the points into an independent float64 buffer before handing them to
  open3d (stock ``--save-world-frame`` segfaults here),
* normals are never estimated,
* defaults follow the cens workflow (250M points, 4096 rays/batch, std_ratio 1.0, world frame on).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
import open3d as o3d
import torch
import tyro

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.exporter.exporter_utils import generate_point_cloud
from nerfstudio.scripts.exporter import ExportPointCloud, validate_pipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE

from cens.cens_datamanager import CENSDataManagerConfig


@dataclass
class CENSExportPointCloud(ExportPointCloud):
    """Export a cens run as a point cloud, sampling rays with a chosen UGS pattern."""

    num_points: int = 250000000
    """Number of points to generate. May result in less if outlier removal is used."""
    reorient_normals: bool = False
    """Reorient point cloud normals based on view direction."""
    normal_method: Literal["open3d", "model_output"] = "open3d"
    """Method to estimate normals with (normals are not estimated by this exporter)."""
    num_rays_per_batch: int = 4096
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    std_ratio: float = 1.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""
    save_world_frame: bool = True
    """If set, saves the point cloud in the same frame as the original dataset. Otherwise, uses the
    scaled and reoriented coordinate space expected by the NeRF models."""
    mask_pattern: Optional[List[int]] = field(default_factory=lambda: [5, 4])
    """UGS pattern used to sample the export rays: indices into pixel_sampler.mask_pattern_name, same format as datamanager.mask_pattern. None keeps the pattern the model was trained with."""
    output_name: str = "u_s250mil_swf_co_mp54.ply"
    """File name of the point cloud, written into output_dir."""

    def _update_sampling_config(self, config: TrainerConfig) -> TrainerConfig:
        """Set the export batch size and sampling pattern before the pipeline (and its DataLoader workers) is built.

        Setting them after eval_setup has no effect: the ray stream is already pickled into the workers.
        Export sampling starts at step 0: the main pattern is walked first, then random draws from the
        support pattern (with a Canny pattern [3, ...] the Canny phase comes first).
        """
        dm = config.pipeline.datamanager
        dm.train_num_rays_per_batch = self.num_rays_per_batch
        if self.mask_pattern is not None:
            if not isinstance(dm, CENSDataManagerConfig):
                CONSOLE.print(
                    "[bold yellow]Warning: not a cens config, mask_pattern is ignored "
                    "(convert old runs with scripts/migrate_config.py)."
                )
            else:
                dm.uniform_grid_sampling = True
                dm.mask_pattern = list(self.mask_pattern)
                names = [dm.pixel_sampler.mask_pattern_name[i] for i in dm.mask_pattern]
                CONSOLE.print(f"Export sampling pattern: {dm.mask_pattern} {names}")
        return config

    def main(self) -> None:
        """Export the point cloud."""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config, update_config_callback=self._update_sampling_config)

        validate_pipeline(self.normal_method, self.normal_output_name, pipeline)

        assert isinstance(pipeline.datamanager, (VanillaDataManager, ParallelDataManager))
        if isinstance(pipeline.datamanager, VanillaDataManager):
            assert pipeline.datamanager.train_pixel_sampler is not None
            pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

            pipeline.datamanager.train_pixel_sampler.config.export_mode = True
            pipeline.datamanager.train_pixel_sampler.kwargs["grid_frac"] = 0.5
        # ParallelDataManager (cens): batch size and pattern are set in _update_sampling_config.

        crop_obb = None
        if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
            crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)

        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            reorient_normals=self.reorient_normals,
            estimate_normals=False,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            crop_obb=crop_obb,
            std_ratio=self.std_ratio,
        )
        if self.save_world_frame:
            # apply the inverse dataparser transform to the point cloud
            points = np.asarray(pcd.points)
            poses = np.eye(4, dtype=np.float32)[None, ...].repeat(points.shape[0], axis=0)[:, :3, :]
            poses[:, :3, 3] = points
            poses = pipeline.datamanager.train_dataparser_outputs.transform_poses_to_original_space(
                torch.from_numpy(poses)
            )
            # Give open3d a fully independent float64 buffer; passing the tensor-backed array segfaults.
            points = poses[:, :3, 3].detach().clone().contiguous().cpu().numpy().astype(np.float64, copy=True)
            pcd.points = o3d.utility.Vector3dVector(points)

        CONSOLE.print(f":cloud: [bold blue]:Generated {pcd}, post-cleanup.")
        CONSOLE.print("Saving Point Cloud...")
        tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        # The legacy PLY writer converts colors to UInt8, let us do the same to save space.
        tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)  # type: ignore
        o3d.t.io.write_point_cloud(str(self.output_dir / self.output_name), tpcd)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")


Commands = {"pointcloud": CENSExportPointCloud}
"""Subcommands of ``ns-export-cens``, same names as stock ``ns-export`` (add e.g. "tsdf" here later)."""


def entrypoint() -> None:
    """Entrypoint for use with pyproject scripts (``ns-export-cens``)."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.extras.subcommand_cli_from_dict(Commands, config=(tyro.conf.FlagConversionOff,)).main()


if __name__ == "__main__":
    entrypoint()
