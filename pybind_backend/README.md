# pybind11 / torch Python interface

Python-callable interface to the CT reconstruction backend (`src/`),
built with `torch.utils.cpp_extension.load` per the course's required
pattern (see `pybindextension/` for the original minimal example this
follows).

## Requirements

- Linux (the C sources `#include <hdf5.h>` and `<omp.h>` unconditionally;
  neither is set up for macOS here — build and run on pool15 / kale).
- Python packages: `torch`, `numpy`, `h5py`.
- `ninja` (torch's JIT build requires it; not always pulled in
  automatically — `pip install ninja` if `from pybind_backend.backend
  import _backend` fails with "Ninja is required").
- HDF5 dev headers/libs and an OpenCL ICD, same as the CLI build
  (see the repo's top-level `README.md` / `Makefile`).

## Usage

```python
from pybind_backend.backend import _backend, KERNEL_DIR

volume = _backend.reconstruct_gpu_opt(
    proj, angles,               # numpy arrays: proj [num_projs,H,W] float32, angles [num_projs] float64
    voxelSize, pixelSize, SDD, SOD,
    Nxz, Ny, W, H, num_projs,
    kernel_dir=KERNEL_DIR,
    epochs=100, subsets=1,      # subsets>1 = OSEM
)
# volume: numpy float32 array, shape (Nxz, Nxz, Ny)
```

Four entry points, mirroring the CLI's `--mode`:

| Function | CLI equivalent |
|---|---|
| `reconstruct_cpu(...)` | `--mode cpu` |
| `reconstruct_gpu_buf(..., kernel_dir=...)` | `--mode gpu-buf` |
| `reconstruct_gpu_img(..., kernel_dir=...)` | `--mode gpu-img` |
| `reconstruct_gpu_opt(..., kernel_dir=..., subsets=...)` | `--mode gpu-opt` |

All four return the reconstructed volume as a numpy `float32` array; none
of them touch HDF5 — read input and write output in Python (see
`example_run.py`).

Demo / reference call site:

```bash
python3 -m pybind_backend.example_run --mode gpu-opt --epochs 100 --subsets 1
```

## Notes

- **First import is slow (~30-60s)** — that's the JIT compile. Subsequent
  imports reuse ninja's cached build (tracked by file mtimes) and are
  fast. If a build ever looks corrupted or stuck, delete the cache and
  retry:
  ```bash
  rm -rf ~/.cache/torch_extensions/ct_recon-*
  ```
- **No `-march=native`, no `-ffast-math`** in this build (unlike the
  Makefile's CLI build) — see the comments in `backend.py` for why: the
  torch JIT cache key doesn't account for CPU architecture or hostname,
  so a native-arch binary built on one lab machine could be silently
  reused on another with a different CPU. Output still agrees with the
  CLI binary at the float32 noise floor (see verification below).
- **`reconstruct_cpu` threading**: uses the same `#pragma omp parallel
  for` as the CLI, but without the Makefile's `OMP_NUM_THREADS` /
  `OMP_PROC_BIND` / `OMP_PLACES` pinning. Set those env vars yourself
  before importing if you want to match `make run-cpu`'s thread pinning.
- **VRAM**: `reconstruct_gpu_opt` checks the requested `subsets` against
  the device's available memory before running, and raises a Python
  exception (with the computed requirement) instead of letting an
  out-of-memory OpenCL allocation kill the process. 512³ volumes need
  `subsets=1` on a 4GB card; higher subset counts need more VRAM than
  is available (`~(3+subsets) × 537MB + 4 × 79MB` at 512³).
- **Ctrl-C** works during a reconstruction — the GIL is released for the
  duration of the C call.

## Verification

Compare wrapper output against the existing CLI binary on identical
input — both run the exact same compiled C/OpenCL code, so results
should agree at the float32 noise floor:

```bash
make run-gpu-opt                                  # produces output_gpu_opt.hdf5
python3 -m pybind_backend.example_run --mode gpu-opt --out output_gpu_opt_py.hdf5
python3 python/validate.py                         # or a direct h5py diff of the two Volume datasets
```
