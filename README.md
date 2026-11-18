# CENS: installation and usage guide

CENS is a [nerfstudio](https://docs.nerf.studio) method plugin. It trains the standard nerfacto model,
but picks the training pixels with **uniform grid sampling (UGS)**: fixed pixel patterns walked in a set
order, optionally preceded by a Canny edge warm-up phase, instead of random pixels.

It installs as a normal Python package next to nerfstudio and adds:

| Command | What it does |
|---|---|
| `ns-train cens` | Train with CENS (all the usual `ns-train` options work) |
| `ns-export-cens pointcloud` | Export a point cloud, sampling the rays with a chosen UGS pattern |

Nerfstudio itself is not modified. Other methods and the stock `ns-*` commands behave as before.

---

## 1. Requirements

- A running nerfstudio container (or environment) with **nerfstudio 1.1.5 or newer**, a CUDA GPU, and
  the usual nerfstudio dependencies (torch, tiny-cuda-nn, open3d).
- The CENS source folder (the one containing `pyproject.toml`).

Check from inside the container:

```bash
ns-train -h | head -5                                                     # nerfstudio works
python3 -c "import nerfstudio, importlib.metadata as m; print(m.version('nerfstudio'))"
```

---

## 2. Installation (no image rebuild)

All steps run **inside the running container**.

### 2.1 Get the source into the container

Pick whichever applies:

- **The source is on a folder already mounted into the container**: nothing to do, note its path.
- **Clone it** (if the container has network access):
  ```bash
  git clone https://github.com/Timeless-H/CENS.git /opt/cens
  ```
- **Copy it in from the host** (run on the host):
  ```bash
  docker cp ./cens <container-name>:/opt/cens
  ```

The examples below use `/opt/cens`; substitute your path.

### 2.2 Install

```bash
pip install --no-deps --no-build-isolation -e /opt/cens
ns-install-cli          # optional: refreshes shell tab completion
```

- `-e` (editable) means edits to the CENS source take effect immediately, without reinstalling.
  Reinstall only if you change `pyproject.toml`.
- `--no-deps` keeps pip from touching the nerfstudio / torch already in the container.
- `--no-build-isolation` uses the container's setuptools, so no download is needed.
- Keep the source folder where it is: an editable install points at it.

### 2.3 Check

```bash
ns-train -h | grep cens               # cens is listed
ns-train cens -h | grep uniform-grid  # the CENS options are shown
ns-export-cens -h                     # the export command exists
```

### 2.4 How long does the install last?

The install lives in the container's filesystem:

| You do | Install still there? |
|---|---|
| `docker stop` / `docker start`, `docker restart`, reboot the host | Yes |
| `docker rm` the container, or it was started with `--rm` | No: rerun step 2.2 in the new container (no rebuild needed) |
| Rebuild the image | No, unless you add the install to the Dockerfile (see below) |

To make it permanent, either snapshot the container (`docker commit <container-name> <new-image:tag>`)
or, if you maintain the image, add to its Dockerfile:

```dockerfile
COPY ./cens /opt/cens
RUN pip install --no-deps --no-build-isolation -e /opt/cens && ns-install-cli
```

### 2.5 Uninstall

```bash
pip uninstall cens
```

---

## 3. Training

### 3.1 Basic run

```bash
ns-train cens --data <path/to/nerfstudio-data>
```

This is nerfacto with three differences:

- UGS is on, with patterns `[4, 1]` (see 3.2),
- masks are ignored for sampling (they are only used as Canny maps, see 3.3),
- the camera optimizer is off.

Everything else (optimizers, schedules, 30k iterations, viewer) matches nerfacto, and the usual options
apply, e.g.:

```bash
ns-train cens --data data/my_scene --max-num-iterations 100000 \
  --pipeline.datamanager.train-num-rays-per-batch 4096 \
  --pipeline.datamanager.dataloader-num-workers 4 \
  --vis viewer+tensorboard \
  nerfstudio-data --downscale-factor 2
```

Run `ns-train cens -h` for the full list.

### 3.2 Sampling patterns

`--pipeline.datamanager.mask-pattern` is a list of indices into the pattern names
(`--pipeline.datamanager.pixel-sampler.mask-pattern-name`). The defaults are:

| Index | Name | Pixels used |
|---|---|---|
| 0 | `mask_101` | checkerboard, ~50% |
| 1 | `mask_010` | the other half of the checkerboard |
| 2 | `mask_all` | every pixel |
| 3 | `mask_canny` | edge pixels of each image (masks are created automatically, see 3.3) |
| 4 | `mask_1000` | ~25%, rows alternate `1000` / `0010` |
| 5 | `mask_0010` | ~25%, the complementary placement to `mask_1000` |

The list has the form `[main, support]`, or `[3, main, support]` for a Canny warm-up:

1. **Canny phase** (only if the list starts with `3`): edge pixels, for
   `--pipeline.datamanager.user-defined-max-iter` steps (default 3000).
2. **Main phase**: the main pattern is walked in a shuffled, fixed order until every image's pattern
   pixels have been used once.
3. **Support phase**: random draws from the support pattern, for the rest of training.

Examples:

```bash
ns-train cens --data data/my_scene --pipeline.datamanager.mask-pattern 4 1      # default
ns-train cens --data data/my_scene --pipeline.datamanager.mask-pattern 3 4 1    # Canny warm-up first
ns-train cens --data data/my_scene --pipeline.datamanager.uniform-grid-sampling False  # plain random sampling
```

Other sampler options:

| Option | Default | Meaning |
|---|---|---|
| `--pipeline.datamanager.pixel-sampler.rng-seed` | 92 | Seed of the pattern order. DataLoader worker *k* uses seed + *k* |
| `--pipeline.datamanager.pixel-sampler.rng-device` | `cuda` | Device the pattern order is generated on. Changing it changes the pixels picked |
| `--pipeline.datamanager.pixel-sampler.ignore-mask` | True | Ignore masks when sampling |

**DataLoader workers.** `--pipeline.datamanager.dataloader-num-workers` greater than 0 is supported. As
in stock nerfstudio, each worker serves its own share of the images, so each worker walks the patterns
over its share.

### 3.3 Masks for the Canny phase (pattern 3)

The Canny phase needs one edge mask per image. **They are created automatically** the first time a
Canny run (`--pipeline.datamanager.mask-pattern 3 ...`) is started on a dataset:

```
data/my_scene/
├── images/                      your images 
├── transforms.json              used by runs without Canny
├── transforms_with_masks.json   transforms.json + a mask_path per frame, used by Canny runs
└── masks/                       frame_00001.png, ... (edge masks)
```

- `transforms.json` is never modified. Runs without Canny keep using it and load no masks.
- `transforms_with_masks.json` is rebuilt from `transforms.json` at the start of every Canny run, so
  later edits to `transforms.json` carry over.
- Only missing masks are generated, so later runs reuse them. **If you replace the images, delete
  `masks/`** so the masks are regenerated.
- With `--downscale-factor N` (in `nerfstudio-data`), the masks go to `masks_N/` and are computed from
  `images_N/`, the images that are actually trained on.
- The saved `config.yml` keeps your original `--data` path. Loading a Canny run again (`ns-export-cens`,
  `ns-viewer`, ...) switches to `transforms_with_masks.json` the same way.
- Masks are 1-bit PNGs, about 3–4% of the size of PNG images (e.g. 0.07 MB per 1024×768 mask).
- Only the `nerfstudio-data` dataparser (`transforms.json` datasets) is supported.

**Datasets whose `transforms.json` already lists masks** (`mask_path` entries, e.g. from an older
version of this tool): those files are used as they are, and any that are missing are generated, also in
runs without Canny. Nerfstudio loads every mask listed in the json, and would otherwise stop with
`FileNotFoundError: .../masks/...`.

To generate the masks ahead of time, without training:

```python
import sys; sys.path.insert(0, "/opt/cens/scripts")
from add_mask_paths import validate_and_setup_masks

validate_and_setup_masks([3, 4, 1], "data/my_scene")                       # full resolution
validate_and_setup_masks([3, 4, 1], "data/my_scene", downscale_factor=2)   # for --downscale-factor 2
```

---

## 4. Point cloud export

```bash
ns-export-cens pointcloud \
  --load-config outputs/<experiment>/cens/<timestamp>/config.yml \
  --output-dir exports/my_scene
```

It takes the same options as stock `ns-export pointcloud`, plus:

| Option | Default | Meaning |
|---|---|---|
| `--mask-pattern` | `5 4` | UGS pattern for the export rays (same format as in training). `None` = the training pattern |
| `--output-name` | `pointcloud_file.ply` | File name inside `--output-dir` |

Defaults that differ from stock: 250M points, 4096 rays per batch, `--std-ratio 1.0`,
`--save-world-frame True`, and normals are not estimated.

Crop to a box with all three of `--obb-center`, `--obb-rotation`, `--obb-scale` (default: no crop):

```bash
ns-export-cens pointcloud --load-config ... --output-dir ... \
  --obb-center 0.0 -0.05 0.10 --obb-rotation 0.30 0.0 -0.06 --obb-scale 0.70 1.55 0.60
```

Notes:

- Use `ns-export-cens` rather than `ns-export pointcloud --save-world-frame`: the stock command can
  segfault in open3d when saving in the world frame. `ns-export-cens` avoids that.
- Export rays follow the same phases as training, starting from step 0.
- `--num-points` is a target: outlier removal and cropping usually leave fewer.

Other `ns-*` commands (`ns-viewer`, `ns-eval`, `ns-render`) work on CENS runs as on any nerfstudio run.

---

## 5. Helper scripts

The `scripts/` folder contains Python versions of the workflows, useful for debugging. Each has
**hard-coded paths in its `__main__` block: edit them before running**.

| Script | Purpose |
|---|---|
| `debug_script.py` | Training run from Python (same as `ns-train cens` with the settings in `reduce_config`) |
| `export_script.py` | Point cloud export from Python (same as `ns-export-cens pointcloud`) |
| `sim_script.py` | Point clouds rendered from simulated cameras, using only the pixels of a pattern |
| `add_mask_paths.py` | Generates the Canny-phase masks ahead of time (training does it automatically, see 3.3) |

Run them by path, e.g. `python3 /opt/cens/scripts/export_script.py`. Set `CENS_DEBUGPY=1` to make a
script wait for a debugger (debugpy) on port 5678.

---

