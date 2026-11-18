"""Prepare the Canny-phase edge masks of a dataset by hand.

``ns-train cens`` does this automatically (see cens/cens_masks.py); this script runs the same step
without training, e.g. to generate the masks ahead of time.
"""

from pathlib import Path
from typing import List, Optional

from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig

from cens.cens_masks import prepare_masks
from cens.cens_pixel_sampler import CENSPixelSamplerConfig


def validate_and_setup_masks(mask_list: List[int], data_dir: str, downscale_factor: Optional[int] = None) -> Path:
    """Generate the missing masks for a run with pattern ``mask_list`` on ``data_dir``.

    Returns the data path such a run reads: ``<data_dir>/transforms_with_masks.json`` for a Canny run on a
    dataset without masks, else ``data_dir``.
    """
    parser_config = NerfstudioDataParserConfig(data=Path(data_dir), downscale_factor=downscale_factor)
    new_data = prepare_masks(parser_config, mask_list, CENSPixelSamplerConfig().mask_pattern_name)
    return new_data if new_data is not None else Path(data_dir)


if __name__ == "__main__":
    from debug_script import reduce_config

    from nerfstudio.configs.method_configs import all_methods

    from cens.cens_datamanager import CENSDataManager

    # Note: reduce_config also saves a config.yml under <tmp_path>/outputs/<experiment>/cens/<timestamp>/.
    selected_config = reduce_config(all_methods["cens"], tmp_path=Path("/wkspace/"))

    # Same step ns-train runs, for the dataset, pattern and downscale factor in the config.
    CENSDataManager._prepare_masks(selected_config.pipeline.datamanager)
    print("Data path for this run:", selected_config.pipeline.datamanager.dataparser.data)
