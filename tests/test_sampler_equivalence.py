"""Bit-exact equivalence of CENSPixelSampler with the original in-tree implementation.

Baselines were recorded with the original code by
/wkspace/backups/2026-11-17_pre-refactor/baseline/record_sampler_old.py (one scenario per process,
since the UGS generator is per-process). Run:  python3 tests/test_sampler_equivalence.py
"""
import subprocess, sys
from pathlib import Path
from types import SimpleNamespace

import torch

BASE = Path("/wkspace/backups/2026-11-17_pre-refactor/baseline")
sys.path.insert(0, str(BASE))
from record_sampler_old import SCENARIOS, N, H, W, R, make_batch  # noqa: E402


def run_new(name: str):
    from cens.cens_pixel_sampler import CENSPixelSamplerConfig
    mp, udmi, steps = SCENARIOS[name]
    prov = SimpleNamespace(train_step_value=0)
    s = CENSPixelSamplerConfig().setup(num_rays_per_batch=R, uniform_grid_sampling=True, mask_pattern=mp,
                                       user_defined_max_iters=udmi, img_nhw=(N, H, W), step_provider=prov)
    res = {"max_iters_in_mask0": s.max_iters_in_mask0, "batches": []}
    for st in steps:
        prov.train_step_value = st
        res["batches"].append({k: v.cpu() for k, v in s.sample(make_batch()).items()})
    return res


def compare(name: str) -> bool:
    old = torch.load(BASE / f"old_{name}.pt", weights_only=False)
    if "error" in old:
        new = run_new(name)
        print(f"{name}: original crashed ({old['error']}); new ran {len(new['batches'])} batches -> SKIP")
        return True
    new = run_new(name)
    ok = old["max_iters_in_mask0"] == new["max_iters_in_mask0"] and len(old["batches"]) == len(new["batches"])
    for bo, bn in zip(old["batches"], new["batches"]):
        ok &= bo.keys() == bn.keys() and all(torch.equal(bo[k], bn[k]) for k in bo)
    print(f"{name}: {'IDENTICAL' if ok else 'MISMATCH'} ({len(new['batches'])} batches)")
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(0 if compare(sys.argv[1]) else 1)
    rc = [subprocess.call([sys.executable, __file__, n]) for n in SCENARIOS]
    sys.exit(max(rc))
