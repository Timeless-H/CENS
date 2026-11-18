"""Edge masks for the CENS Canny phase: generation and dataset preparation.

``prepare_masks`` is called by ``CENSDataManager`` before the dataparser reads the transforms file:

* Canny runs (pattern 3) on a dataset without ``mask_path`` entries get ``transforms_with_masks.json``
  (a copy of the transforms file plus ``mask_path`` entries); ``transforms.json`` is never modified.
* Every mask file the chosen transforms file refers to that does not exist yet is generated, at the path
  (and resolution) nerfstudio will read it from, from the image nerfstudio will read.

Masks are written as 1-bit palette PNGs (index 0 = black, 1 = white; ~3.5% of a PNG image). (A PIL
mode-"1" PNG would be decoded wrongly by nerfstudio's ``pil_to_numpy``, hence the palette.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.utils.rich_utils import CONSOLE

from cens.cens_mask_patterns import canny_raw_magnitude

MASKS_SUFFIX = "_with_masks"
"""``transforms.json`` -> ``transforms_with_masks.json``."""


def generate_mask(image_path: Path, mask_path: Path) -> None:
    """Write the Canny edge mask of ``image_path`` to ``mask_path`` as a 1-bit palette PNG."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image {image_path}")
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.fromarray((canny_raw_magnitude(img) > 0).astype(np.uint8)).convert("P")
    mask.putpalette([0, 0, 0, 255, 255, 255])
    mask.save(mask_path, format="PNG", bits=1, optimize=True)


def _transforms_paths(data: Path) -> tuple:
    """(transforms file, data dir) as the nerfstudio dataparser resolves them."""
    if data.suffix == ".json":
        return data, data.parent
    return data / "transforms.json", data


def _with_masks_path(transforms_file: Path) -> Path:
    return transforms_file.with_name(transforms_file.stem + MASKS_SUFFIX + ".json")


def _default_mask_path(file_path: str) -> str:
    """``images/frame_00001.jpg`` -> ``masks/frame_00001.png``."""
    return str(Path("masks") / (Path(file_path).stem + ".png"))


def write_transforms_with_masks(transforms_file: Path) -> Path:
    """(Re)write ``<transforms>_with_masks.json``: the transforms file plus a ``mask_path`` per frame."""
    meta = json.loads(transforms_file.read_text())
    for frame in meta["frames"]:
        frame["mask_path"] = _default_mask_path(frame["file_path"])
    out = _with_masks_path(transforms_file)
    out.write_text(json.dumps(meta, indent=4))
    return out


def generate_missing_masks(dataparser_config: NerfstudioDataParserConfig, transforms_file: Path) -> int:
    """Generate every mask referenced by ``transforms_file`` that does not exist yet. Returns the count.

    Paths are resolved with the dataparser's own logic, so with ``--downscale-factor N`` the masks go to
    ``masks_N/`` and are computed from ``images_N/``, exactly the files nerfstudio loads.
    """
    meta = json.loads(transforms_file.read_text())
    data_dir = transforms_file.parent
    parser = dataparser_config.setup()
    todo: List[tuple] = []
    for frame in meta["frames"]:
        if "mask_path" not in frame:
            continue
        # Resolve the image first: in auto-downscale mode the dataparser picks the factor from it.
        image_file = parser._get_fname(Path(frame["file_path"]), data_dir)
        mask_file = parser._get_fname(Path(frame["mask_path"]), data_dir, downsample_folder_prefix="masks_")
        if not mask_file.exists():
            todo.append((image_file, mask_file))
    if todo:
        CONSOLE.print(f"[bold]CENS: generating {len(todo)} missing edge masks in {todo[0][1].parent} ...")
        for image_file, mask_file in todo:
            generate_mask(image_file, mask_file)
    return len(todo)


def prepare_masks(dataparser_config, mask_pattern: List[int], mask_pattern_name: List[str]) -> Optional[Path]:
    """Make sure the masks this run will load exist. Returns the data path to use if it must change, else None."""
    if not isinstance(dataparser_config, NerfstudioDataParserConfig) or dataparser_config.data is None:
        if any(mask_pattern_name[i] == "mask_canny" for i in mask_pattern):
            CONSOLE.print("[bold yellow]CENS: masks are only prepared for the nerfstudio dataparser; skipping.")
        return None
    transforms_file, _ = _transforms_paths(Path(dataparser_config.data))
    if not transforms_file.exists():
        return None  # let the dataparser report it
    frames: List[Dict] = json.loads(transforms_file.read_text())["frames"]
    uses_canny = any(mask_pattern_name[i] == "mask_canny" for i in mask_pattern)

    new_data: Optional[Path] = None
    if uses_canny and not any("mask_path" in f for f in frames):
        transforms_file = write_transforms_with_masks(transforms_file)
        new_data = transforms_file
        CONSOLE.print(f"[bold]CENS: Canny phase uses {transforms_file} (transforms.json is unchanged).")
    if any("mask_path" in f for f in json.loads(transforms_file.read_text())["frames"]):
        generate_missing_masks(dataparser_config, transforms_file)
    return new_data
