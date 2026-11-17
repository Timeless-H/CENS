"""Convert a config.yml written by the old patched-nerfstudio CENS (method 'nerfacto' + UGS) to the cens plugin.

Usage: python3 migrate_config.py <old config.yml> [<out.yml>]   (default out: <old>_cens.yml next to it)

The model stays NerfactoModel (identical weights / state dict); only the datamanager and pixel
sampler are switched to the CENS classes, keeping uniform_grid_sampling / mask_pattern /
user_defined_max_iter. The original file is not modified.
"""
import re
import sys
from pathlib import Path

import yaml

from cens.cens_pixel_sampler import CENSPixelSamplerConfig

OLD_DM = "nerfstudio.data.datamanagers.parallel_datamanager.ParallelDataManager"


def migrate(text: str) -> str:
    if "cens.cens_datamanager" in text:
        raise SystemExit("already a cens config")
    if f"{OLD_DM}Config" not in text or "uniform_grid_sampling" not in text:
        raise SystemExit("not an old CENS (ParallelDataManager + UGS) config")
    text = text.replace(f"!!python/object:{OLD_DM}Config", "!!python/object:cens.cens_datamanager.CENSDataManagerConfig")
    text = text.replace(f"!!python/name:{OLD_DM} ''", "!!python/name:cens.cens_datamanager.CENSDataManager ''")
    # Replace the whole pixel_sampler block (it pickled a torch.Generator) with a fresh CENS sampler config.
    block = re.compile(r"^    pixel_sampler: !!python/object:nerfstudio\.data\.pixel_samplers\.PixelSamplerConfig\n(?:^      .*\n|^\s*\n)*", re.M)
    if not block.search(text):
        raise SystemExit("pixel_sampler block not found")
    new = yaml.dump(CENSPixelSamplerConfig(), default_flow_style=False)
    new = "    pixel_sampler: " + new.replace("\n", "\n      ").rstrip() + "\n"
    return block.sub(lambda _: new, text, count=1)


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + "_cens.yml")
    dst.write_text(migrate(src.read_text()))
    print(f"wrote {dst}")
