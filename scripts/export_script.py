"""Export a cens run from Python; CLI equivalent: ``ns-export-cens pointcloud --load-config ... --output-dir ...``.

Set CENS_DEBUGPY=1 to wait for a debugger on port 5678.
"""

import os

if os.environ.get("CENS_DEBUGPY") == "1":
    import debugpy

    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger...")
    debugpy.wait_for_client()

from pathlib import Path

from nerfstudio.utils.rich_utils import CONSOLE

from cens.cens_exporter import CENSExportPointCloud

if __name__ == "__main__":
    base_dir = Path("/wkspace/outputs/jamtal_t1/nerfacto/2026-04-04_132417")

    exporter = CENSExportPointCloud(
        output_dir=base_dir / "pointcloud",
        load_config=base_dir / "config.yml",
        # Jamtal crop box (alternatives tried: center -0.37, 0.0, 0.0; scale 1.0, 1.15, 1.15)
        obb_center=(0.00, -0.05, 0.10),
        obb_rotation=(0.30, 0.0, -0.06),
        obb_scale=(0.70, 1.55, 0.60),
        mask_pattern=[5, 4],
        output_name="u_s250mil_swf_co_mp54.ply",  # mp: mask pattern; swf: save world frame; std ratio 1.0
    )

    CONSOLE.print("Starting export process... :crossed_fingers:")
    exporter.main()
    CONSOLE.print(":thumbs_up: Export process completed. :waving_hand:")
