import os

if os.environ.get("CENS_DEBUGPY") == "1":  # set CENS_DEBUGPY=1 to wait for a debugger on port 5678
    import debugpy

    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger...")
    debugpy.wait_for_client()

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing_extensions import Literal
from typing import List, Optional, Tuple
import numpy as np
import open3d as o3d
import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.exporter.exporter_utils import generate_point_cloud
from nerfstudio.utils.rich_utils import CONSOLE

from cens.cens_datamanager import CENSDataManagerConfig

@dataclass
class ExporterConfig:
    """Configuration for the export script."""

    output_dir: Path
    """Path to the output directory where exported data will be saved."""
    load_config: Path
    """Path to the YML configuration file to load."""


def validate_pipeline(normal_method: str, normal_output_name: str, pipeline: Pipeline) -> None:
    """Check if the pipeline is valid for the export process.

    Args:
        normal_method: Method to estimate normals with. Either "open3d" or "model_output".
        normal_output_name: Name of the normal output.
        pipeline: Pipeline to evaluate with.
    """
    if normal_method == "model_output":
        CONSOLE.print("Checking that the pipeline has a normal output.")
        origins = torch.zeros((1, 3), device=pipeline.device)
        directions = torch.ones_like(origins)
        pixel_area = torch.ones_like(origins[..., :1])
        camera_indices = torch.zeros_like(origins[..., :1])
        metadata = {"directions_norm": torch.linalg.vector_norm(directions, dim=-1, keepdim=True)}
        ray_bundle = RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=pixel_area,
            camera_indices=camera_indices,
            metadata=metadata,
        )
        outputs = pipeline.model(ray_bundle)
        if normal_output_name not in outputs:
            CONSOLE.print(f"[bold yellow]Warning: Normal output '{normal_output_name}' not found in pipeline outputs.")
            CONSOLE.print(f"Available outputs: {list(outputs.keys())}")
            CONSOLE.print(
                "[bold yellow]Warning: Please train a model with normals "
                "(e.g., nerfacto with predicted normals turned on)."
            )
            CONSOLE.print("[bold yellow]Warning: Or change --normal-method")
            CONSOLE.print("[bold yellow]Exiting early.")
            sys.exit(1)


@dataclass
class ExportPointCloud(ExporterConfig):
    """Export NeRF as a point cloud."""

    num_points: int = 250000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = False
    """Reorient point cloud normals based on view direction."""
    normal_method: Literal["open3d", "model_output"] = "open3d"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    depth_output_name: str = "depth" # vanilla_nerf: "depth_fine"
    """Name of the depth output."""
    rgb_output_name: str = "rgb" # vanilla_nerf: "rgb_fine"
    """Name of the RGB output."""
    obb_center: Optional[Tuple[float, float, float]] = (0.00, -0.05, 0.10) # -0.37, 0.0, 0.0; None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = (0.30, 0.0, -0.06) # None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = (0.70, 1.55, 0.60) # 1.0, 1.15, 1.15; None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 4096 #32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    mask_pattern: Optional[List[int]] = field(default_factory=lambda: [5, 4])
    """UGS pattern used to sample the export rays (indices into pixel_sampler.mask_pattern_name, same format
    as datamanager.mask_pattern). None keeps the pattern the model was trained with."""
    std_ratio: float = 1.0          # CHANGED BY Hope Akwensi 08.04.2026
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""
    save_world_frame: bool = True
    """If set, saves the point cloud in the same frame as the original dataset. Otherwise, uses the
    scaled and reoriented coordinate space expected by the NeRF models."""

    def _update_sampling_config(self, config):
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

    def main(self):
        """export the point cloud."""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config, update_config_callback=self._update_sampling_config)

        validate_pipeline(self.normal_method, self.normal_output_name, pipeline)

        assert isinstance(pipeline.datamanager, (VanillaDataManager, ParallelDataManager), )
        if isinstance(pipeline.datamanager, VanillaDataManager):
            assert pipeline.datamanager.train_pixel_sampler is not None
            pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch # increase batchsize to speed up process at the expense of memory

            pipeline.datamanager.train_pixel_sampler.config.export_mode = True
            pipeline.datamanager.train_pixel_sampler.kwargs["grid_frac"] = 0.5
        # ParallelDataManager (cens): batch size and pattern are set in _update_sampling_config.

        # whether the normals should be estimated based on the point cloud (via open3d)
        estimate_normals = False # self.normal_method == "open3d"
        crop_obb = None
        if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
            crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)

        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            reorient_normals=self.reorient_normals,
            estimate_normals=estimate_normals,
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
            
            """Solving the Segmentation faukt (core dumped) issue between open3d and pytorch tensors via numpy. i.e.,
               Creating a fully independent numpy array"""
            #points = poses[:, :3, 3].numpy()
            points = (                              # MODIFIED BY Hope Akwensi 08.04.2026: to solve segmentation fault issue between open3d and pytorch tensors
                poses[:, :3, 3]
                .detach()      # removes autograd tracking
                .clone()       # ensures the memory is not shared with any other tensor
                .contiguous()  # ensures row-major layout
                .cpu()         # ensures CPU memory
                .numpy()       # produces a NumPy array
                .astype(np.float64, copy=True)  # ensures Open3D gets its own buffer
            )

            pcd.points = o3d.utility.Vector3dVector(points)

        CONSOLE.print(f":cloud: [bold blue]:Generated {pcd}, post-cleanup.")
        CONSOLE.print("Saving Point Cloud...")
        tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        # The legacy PLY writer converts colors to UInt8,
        # let us do the same to save space.
        tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)  # type: ignore
        o3d.t.io.write_point_cloud(str(self.output_dir / "u_s250mil_swf_co_mp54.ply"), tpcd) # r_s25mil_2x_swf_co.ply; mp: mask pattern; swf: save world frame; std ratio of 1.0
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")
        

if __name__ == "__main__":
    base_dir = Path("/wkspace/outputs/jamtal_t1/nerfacto/2026-04-04_132417")

    exporter = ExportPointCloud(
        output_dir=base_dir/"pointcloud",
        load_config=base_dir/"config.yml",
    )

    CONSOLE.print("Starting export process... :crossed_fingers:")
    exporter.main()
    CONSOLE.print(":thumbs_up: Export process completed. :waving_hand:")