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
`run.py`).

Demo / reference call site:

```bash
python3 -m pybind_backend.run --mode gpu-opt --epochs 100 --subsets 1
```

## Notes

- **First import is slow (~30-60s)** — that's the JIT compile. Subsequent
  imports reuse ninja's cached build (tracked by file mtimes) and are
  fast. If a build ever looks corrupted or stuck, delete the cache and
  retry:
  ```bash
  rm -rf ~/.cache/torch_extensions/ct_recon-*
  ```
- **`-march=native` and `-ffast-math`** match the Makefile's CLI build.
  The JIT build directory is namespaced per-host (see `backend.py`), so
  a native-arch binary is never shared across machines — safe to use.
  Without `-march=native`, `reconstruct_cpu` measured ~1.8x slower on
  kale (7.2s/epoch vs the CLI's 3.9s/epoch, same OMP env on both
  sides); with it, timings match. Output agrees with the CLI binary at
  the float32 noise floor (see verification below).
- **CPU thread pinning**: `backend.py` sets `OMP_NUM_THREADS` /
  `OMP_PROC_BIND=close` / `OMP_PLACES=cores` at import time (matching
  the Makefile's `run-cpu` target) unless already set in the
  environment. Without pinning, `reconstruct_cpu`'s per-epoch time was
  observed climbing steadily on kale (5.5s → 14s+ over ~25 epochs) as
  OpenMP threads drifted across cores; with pinning it stays flat.
- **`reconstruct_cpu` threading**: uses the same `#pragma omp parallel
  for` as the CLI, but without the Makefile's `OMP_NUM_THREADS` /
  `OMP_PROC_BIND` / `OMP_PLACES` pinning. Set those env vars yourself
  before importing if you want to match `make run-cpu`'s thread pinning.
- **Flush-to-zero / denormals-are-zero**: `-ffast-math` sets FTZ/DAZ via
  `crtfastmath.o`, which is only linked into a *main executable* — the
  CLI's `main.c` gets it for free, but this extension's main executable
  is `python3`, built without `-ffast-math`, so compiling the `.so` with
  the flag alone did not set MXCSR here. Symptom on kale:
  `reconstruct_cpu`'s `fp_cpu` time was flat for ~20 epochs, climbed to
  ~2.9x by epoch ~30, then slowly recovered — background voxels decaying
  through the float32 denormal range as MLEM converges, at the ~100x
  per-op cost of unflushed denormal arithmetic. `bp_cpu` never showed it
  (it only ever writes `volume`, never reads it back). Fixed by setting
  FTZ/DAZ explicitly in `ct_recon_bindings.cpp`'s `PYBIND11_MODULE` init
  (`_MM_SET_FLUSH_ZERO_MODE` / `_MM_SET_DENORMALS_ZERO_MODE`); confirmed
  OpenMP worker threads inherit MXCSR from the thread that set it, so one
  call at import time is sufficient. Confirmed on kale: `fp_cpu` now
  holds flat for 40+ epochs, and output is bit-identical to the CLI
  (`MSE: 0.0`, `max abs diff: 0.0`) — the earlier ~1e-79 noise floor was
  itself residual denormal arithmetic, now gone entirely.
- **Env vars must be set before `import torch`, not just before the C
  call**: `backend.py` sets `OMP_NUM_THREADS`/`OMP_PROC_BIND`/`OMP_PLACES`
  via `os.environ.setdefault(...)` above the `torch.utils.cpp_extension`
  import. This ordering matters, not just the values — torch's own C++
  init reads/caches OpenMP configuration once at import time, and GOMP's
  runtime is shared in-process with torch's. Setting the vars *after*
  `import torch` (even via `setdefault`, even moments later) left
  `reconstruct_cpu` a flat, reproducible ~13% slower than the CLI on both
  `fp` and `bp` (3.18s/1.11s vs 2.84s/0.94s per epoch) despite identical
  values ending up in `os.environ` either way. An identical in-process
  call with the vars set before `import torch` matched the CLI exactly.
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
python3 -m pybind_backend.run --mode gpu-opt --out output_gpu_opt_py.hdf5
python3 python/validate.py                         # or a direct h5py diff of the two Volume datasets
```
