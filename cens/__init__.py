"""CENS nerfstudio method plugin."""

import os

# This torch version defaults torch.load(weights_only=True), which cannot load nerfstudio checkpoints
# (ns-eval/ns-export/ns-viewer, resume). Formerly patched in nerfstudio/utils/eval_utils.py.
# Only affects call sites that don't pass weights_only explicitly; an existing env value wins.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
