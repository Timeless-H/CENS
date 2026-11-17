import os

if os.environ.get("CENS_DEBUGPY") == "1":  # set CENS_DEBUGPY=1 to wait for a debugger on port 5678
    import debugpy

    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger...")
    debugpy.wait_for_client()

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing_extensions import Literal
from typing import Optional, Tuple

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.cameras.cameras import Cameras, CameraType
from rich.progress import (Progress, TextColumn, BarColumn, TimeRemainingColumn)

from cens.cens_mask_patterns import gen_active_indices_for_pattern
from cens.cens_model import get_pattern_outputs_for_camera_ray_bundle
# from my_utils import (delaunay_25d, sample_pn_from_mesh, pn_to_poses, camera_data_extract)


@dataclass
class SimExporterConfig:
    """Configuration for the export script."""

    output_dir: Path
    """Path to the output directory where exported data will be saved."""
    model_config: Path
    """Path to the YML configuration file to load for the NeRF model."""
    sim_config: Path
    """Path to the YML configuration file to load for the cameras to use for simulation."""

@dataclass
class ExportSimPointCloud(SimExporterConfig):
    """Export NeRF as a point cloud."""

    num_points: int = 25000000 # 425mil @ 10% 2730x4096 pixels
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    # reorient_normals: bool = True
    # """Reorient point cloud normals based on view direction."""
    # normal_method: Literal["open3d", "model_output"] = "open3d"
    # """Method to estimate normals with."""
    # normal_output_name: str = "normals"
    # """Name of the normal output."""
    depth_output_name: str = "depth" # vanilla_nerf: "depth_fine"
    """Name of the depth output."""
    rgb_output_name: str = "rgb" # vanilla_nerf: "rgb_fine"
    """Name of the RGB output."""
    obb_center: Optional[Tuple[float, float, float]] =(0.00, -0.05, 0.10) # None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = (0.30, 0.00, -0.06) # None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = (0.70, 1.55, 0.60) # None
    """Scale of the oriented bounding box along each axis."""
    std_ratio: float = 1.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""
    save_world_frame: bool = True
    """If set, saves the point cloud in the same frame as the original dataset. Otherwise, uses the
    scaled and reoriented coordinate space expected by the NeRF models."""
    ugs: bool = True
    """Whether to use uniform grid sampling for ray bundles instead of the default full image or 
    random sampling. UGS Only applicable if the model supports it."""
    mask_patterns: list[str] = field(default_factory=lambda: ["mask_101", "mask_010", "mask_all", "mask_canny", "mask_1000", "mask_0010", "mask_arrow10"])
    """The mask patterns available to use for uniform grid sampling."""
    selected_mask: str = "mask_arrow10"
    """The mask pattern to use for uniform grid sampling, chosen from the mask_patterns list. Only applicable if ugs is set to True."""
    g: torch.Generator = torch.Generator(device=torch.device("cuda")).manual_seed(92)

    def main(self):
        """export the point cloud."""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, m_pipeline, _, _ = eval_setup(self.model_config)
        _, s_pipeline, _, _ = eval_setup(self.sim_config)

        assert isinstance(m_pipeline.datamanager, ParallelDataManager)

        # === Define new camera poses ===
        tr_cameras = s_pipeline.datamanager.train_dataset.cameras
        val_cameras = s_pipeline.datamanager.eval_dataset.cameras

        all_cam_poses = torch.cat([tr_cameras.camera_to_worlds, val_cameras.camera_to_worlds], dim=0)
        all_cam_fxs = torch.cat([tr_cameras.fx, val_cameras.fx], dim=0)
        all_cam_fys = torch.cat([tr_cameras.fy, val_cameras.fy], dim=0)
        all_cam_cxs = torch.cat([tr_cameras.cx, val_cameras.cx], dim=0)
        all_cam_cys = torch.cat([tr_cameras.cy, val_cameras.cy], dim=0)
        all_cam_widths = torch.cat([tr_cameras.width, val_cameras.width], dim=0)
        all_cam_heights = torch.cat([tr_cameras.height, val_cameras.height], dim=0)
        #TODO: will a basic transformation between the sim and model camera spaces be necessary here?
        # origins = all_camera_poses[..., :3, -1]

        # # flight plane surface generation
        # faces, proj_points, nmls, dists = delaunay_25d(origins)

        # # === Sample points and normals from the mesh ===
        # new_origins, new_normals = sample_pn_from_mesh(origins, faces, num_samples=origins.shape[0]*4)
        # new_poses = pn_to_poses(new_origins, new_normals)
        # new_poses = new_poses[:, :3, :]

        mask_p = gen_active_indices_for_pattern(all_cam_heights[0].item(), all_cam_widths[0].item(), self.selected_mask, g=self.g) # mask_0010

        crop_obb = None
        if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
            crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
        # cameras_params = camera_data_extract(tr_cameras, val_cameras)

        new_cameras = Cameras(
            camera_to_worlds=all_cam_poses,
            fx=all_cam_fxs,
            fy=all_cam_fys,
            cx=all_cam_cxs,
            cy=all_cam_cys,
            width=all_cam_widths,
            height=all_cam_heights,
            camera_type=CameraType.PERSPECTIVE
        )

        progress = Progress(
            TextColumn(":cloud: Generating Simulated Epoch Point Cloud :cloud:"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn()
        )

        with progress as progress_bar:
            task = progress_bar.add_task("Generating Point Cloud", total=len(new_cameras))

            points, rgbs = [], []
            normals, view_directions = [], []
            for c in range(len(new_cameras)):
                cam = new_cameras[c:c+1]
                # === Generate ray bundle for the entire image ===
                ray_bundle = cam.generate_rays(camera_indices=0)
                ray_bundle = ray_bundle.to(m_pipeline.device)

                # === bundle splitting is done within "get_outputs_for_camera_ray_bundle" ===
                # === Get model outputs for the ray_bundle ===
                with torch.no_grad():
                    if self.ugs: # mask ray bundle with sampling pattern
                        outputs = get_pattern_outputs_for_camera_ray_bundle(m_pipeline.model, ray_bundle, mask_p)
                    else: # get full/random sampling outputs for ray bundle
                        outputs = m_pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)
                rgba = m_pipeline.model.get_rgba_image(outputs, self.rgb_output_name)
                depth = outputs[self.depth_output_name]

                # === Reducing memory by limiting everything to active rays ===
                rb = ray_bundle.flatten()[mask_p]
                rgba = rgba.flatten(0, 1)[mask_p]
                depth = depth.flatten()[mask_p]

                # point = ray_bundle.origins + ray_bundle.directions * depth
                point = rb.origins + rb.directions * depth.unsqueeze(-1)
                view_direction = rb.directions

                # Filter points with opacity lower than 0.5
                mask = rgba[..., -1] > 0.5
                point = point[mask]
                view_direction = view_direction[mask]
                rgb = rgba[mask][..., :3]

                if crop_obb is not None:
                    mask = crop_obb.within(point)
                    point = point[mask]
                    rgb = rgb[mask]
                    view_direction = view_direction[mask]

                points.append(point)
                rgbs.append(rgb)
                view_directions.append(view_direction)

                progress_bar.advance(task, advance=1)

        points = torch.cat(points, dim=0)
        rgbs = torch.cat(rgbs, dim=0)
        view_directions = torch.cat(view_directions, dim=0).cpu()

        # === Post processing ===
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.double().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(rgbs.double().cpu().numpy())

        ind = None
        if self.remove_outliers:
            CONSOLE.print("Cleaning Point Cloud")
            pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=self.std_ratio)
            print("\033[A\033[A")
            CONSOLE.print("[bold green]:white_check_mark: Cleaning Point Cloud")
            if ind is not None:
                view_directions = view_directions[ind]

        if self.save_world_frame:
            # apply the inverse dataparser transform to the point cloud
            points = np.asarray(pcd.points)
            poses = np.eye(4, dtype=np.float32)[None, ...].repeat(points.shape[0], axis=0)[:, :3, :]
            poses[:, :3, 3] = points
            poses = m_pipeline.datamanager.train_dataparser_outputs.transform_poses_to_original_space(  #TODO: should this be sim or model dataparser_outputs transform?
                torch.from_numpy(poses)
            )
            # points = poses[:, :3, 3].numpy()
            points = (                              # MODIFIED BY Hope Akwensi 08.04.2026: to solve segmentation fault issue between open3d and pytorch tensors
                poses[:, :3, 3]
                .detach()      # removes autograd tracking
                .clone()       # ensures the memory is not shared with any other tensor
                .contiguous()  # ensures row-major layout
                .cpu()         # ensures CPU memory
                .numpy()       # produces a NumPy array
                .astype(np.float64, copy=True)
            )
            pcd.points = o3d.utility.Vector3dVector(points)

        torch.cuda.empty_cache()

        CONSOLE.print(f":cloud: [bold blue]:Generated {pcd}, post-cleanup.")
        CONSOLE.print("Saving Point Cloud...")
        tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        # The legacy PLY writer converts colors to UInt8,
        # let us do the same to save space.
        tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)  # type: ignore
        o3d.t.io.write_point_cloud(str(self.output_dir / "u_sim_mp10_swf_co_mpt.ply"), tpcd) # r_s25mil_2x_swf_co.ply
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")



if __name__ == "__main__":
    sim_cam_dir = Path("/wkspace/outputs/jamtal_t1/nerfacto/2026-04-04_132417")
    model_dir = Path("/wkspace/outputs/jamtal_t2/nerfacto/2026-04-05_112845")

    exporter = ExportSimPointCloud(
        output_dir=model_dir/"sim_pointcloud",
        model_config=model_dir/"config.yml",
        sim_config=sim_cam_dir/"config.yml"
    )
    
    CONSOLE.print("Starting export process... :crossed_fingers:")
    exporter.main()
    CONSOLE.print(":thumbs_up: Export process completed. :waving_hand:")