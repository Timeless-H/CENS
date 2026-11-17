"""Mask patterns and edge-mask helpers used by CENS uniform grid sampling (UGS).

Moved verbatim (logic-wise) from the former in-tree ``nerfstudio/data/mask_patterns.py``.
"""

from typing import Callable, Dict, Optional, Union

import cv2
import numpy as np
import torch


# --- Mask pattern functions --- #
def mask_101_pattern(H: int, W: int, device: Union[torch.device, str]) -> torch.Tensor:
    """Checkerboard, ~50% active, (0, 0) active."""
    rows = torch.arange(H, device=device).unsqueeze(1)
    cols = torch.arange(W, device=device).unsqueeze(0)
    return ((rows + cols) % 2 == 0).to(torch.uint8)


def mask_010_pattern(H: int, W: int, device: Union[torch.device, str]) -> torch.Tensor:
    """Complement of ``mask_101``."""
    return 1 - mask_101_pattern(H, W, device)


def mask_all_pattern(H: int, W: int, device: Union[torch.device, str]) -> torch.Tensor:
    """All pixels active."""
    return torch.ones((H, W), dtype=torch.uint8, device=device)


def mask_1000_pattern(H: int, W: int, device: Union[torch.device, str]) -> torch.Tensor:
    """~25% active: rows alternate ``1000`` / ``0010``."""
    pattern = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=torch.uint8, device=device)
    tiles_h = (H + 1) // 2
    tiles_w = (W + 3) // 4
    return pattern.repeat(tiles_h, tiles_w)[:H, :W]


def mask_0010_pattern(H: int, W: int, device: Union[torch.device, str]) -> torch.Tensor:
    """~25% active, complementary placement to ``mask_1000``."""
    pattern = torch.tensor([[0, 0, 1, 0], [1, 0, 0, 0]], dtype=torch.uint8, device=device)
    tiles_h = (H + 1) // 2
    tiles_w = (W + 3) // 4
    return pattern.repeat(tiles_h, tiles_w)[:H, :W]


def mask_arrow10_pattern(H: int, W: int, device: Union[torch.device, str]) -> torch.Tensor:
    """Sparse 10x10 'arrow' tiles, alternating tiles A/B vertically."""
    A = torch.tensor(
        [
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.uint8,
        device=device,
    )
    B = torch.tensor(
        [
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.uint8,
        device=device,
    )
    tile_h, tile_w = 10, 10
    tiles_h = (H + tile_h - 1) // tile_h
    tiles_w = (W + tile_w - 1) // tile_w
    vertical_stack = torch.cat([A if (i % 2 == 0) else B for i in range(tiles_h)], dim=0)
    return vertical_stack.repeat(1, tiles_w)[:H, :W]


PATTERN_FUNC_MAP: Dict[str, Callable[..., torch.Tensor]] = {
    "mask_101": mask_101_pattern,
    "mask_010": mask_010_pattern,
    "mask_all": mask_all_pattern,
    "mask_1000": mask_1000_pattern,
    "mask_0010": mask_0010_pattern,
    "mask_arrow10": mask_arrow10_pattern,
}
"""Pattern name -> generator. ``mask_canny`` is not here: it comes from the per-image masks."""


# --- Index helpers --- #
def get_flat_active_indices(mask: torch.Tensor, W: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Shuffled flat indices (r * W + c) of the active pixels (mask != 0)."""
    active_rc = torch.nonzero(mask, as_tuple=False)
    active = active_rc[:, 0] * W + active_rc[:, 1]
    idx = torch.randperm(active.numel(), device=active.device, generator=generator)
    return active[idx]


def flat_index_to_2d_coords(flat_indices: torch.Tensor, W: int, H: int, img_idx: int) -> torch.Tensor:
    """Flat indices (N,) -> (N, 3) coordinates ``(img_idx, row, col)``."""
    img_idxs = torch.empty((flat_indices.numel(), 3), dtype=torch.long, device=flat_indices.device)
    img_idxs[:, 0] = img_idx
    img_idxs[:, 1] = flat_indices // W
    img_idxs[:, 2] = flat_indices % W
    return img_idxs


def gen_active_indices_for_pattern(H: int, W: int, pattern_name: str, g: torch.Generator) -> torch.Tensor:
    """Shuffled flat active indices for a named pattern, generated on ``g.device`` (consumes ``g``)."""
    if pattern_name not in PATTERN_FUNC_MAP:
        raise ValueError(f"Pattern '{pattern_name}' is not recognized. Available patterns: {list(PATTERN_FUNC_MAP)}")
    mask = PATTERN_FUNC_MAP[pattern_name](H, W, device=g.device)
    return get_flat_active_indices(mask, W, g)


# --- Edge masks (used offline to create 'mask_canny' masks, see scripts/add_mask_paths.py) --- #
def sobel_high_freq_mask(img: np.ndarray) -> np.ndarray:
    """Sobel magnitude, kept where >= mid-range threshold."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_disp = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thresh = (mag_disp.min() + mag_disp.max()) / 2
    return (mag_disp >= thresh) * mag_disp


def canny_raw_magnitude(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Canny-style edge mask (uint8 0/255): Gaussian blur, Sobel, NMS, single low threshold.

    The threshold drops from 10% to 6% of the max NMS magnitude when fewer than 125000 edge pixels are found.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if (img.ndim == 3 and img.shape[2] == 3) else img
    blurred = cv2.GaussianBlur(gray, (5, 5), sigma)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    direction = np.arctan2(sobely, sobelx)
    nms_magnitude = non_max_suppression(magnitude, direction)

    edges = (nms_magnitude > 0.1 * np.max(nms_magnitude)).astype(np.uint8) * 255
    if np.count_nonzero(edges) < 125000:
        edges = (nms_magnitude > 0.06 * np.max(nms_magnitude)).astype(np.uint8) * 255
    return edges


def non_max_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Thin edges by keeping local maxima along the gradient direction."""
    rows, cols = magnitude.shape
    nms = np.zeros_like(magnitude)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = (direction[i, j] * 180 / np.pi) % 180
            if 0 <= angle < 22.5 or 157.5 <= angle <= 180:
                neighbors = [magnitude[i, j - 1], magnitude[i, j + 1]]
            elif 22.5 <= angle < 67.5:
                neighbors = [magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]]
            elif 67.5 <= angle < 112.5:
                neighbors = [magnitude[i - 1, j], magnitude[i + 1, j]]
            else:
                neighbors = [magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]]
            if magnitude[i, j] >= max(neighbors):
                nms[i, j] = magnitude[i, j]
    return nms
