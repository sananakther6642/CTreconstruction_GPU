# CT Volume Reconstruction — GPU Lab Project

Cone-beam CT reconstruction using iterative MLEM. CPU (OpenMP) and GPU
(OpenCL) implementations, 256³ and 512³ datasets. Packaged as a
pybind11/torch Python interface (`backend.py`, `run.py`) per the
course's required pattern — wraps the same compiled C/OpenCL sources
(`src/`) as the CLI binary, no duplicated logic.

## Requirements

- Linux (the C sources `#include <hdf5.h>` and `<omp.h>` unconditionally;
  neither is set up for macOS here — build and run on pool15 / kale).
- Python packages: `torch`, `numpy`, `h5py`.
- `ninja` (torch's JIT build requires it; not always pulled in
  automatically — `pip install ninja` if `from backend import _backend`
  fails with "Ninja is required").
- HDF5 dev headers/libs and an OpenCL ICD:
  ```bash
  sudo apt install libhdf5-dev ocl-icd-opencl-dev opencl-headers
  ```

No separate build step — `backend.py` JIT-compiles on first import.

## Usage

```python
from backend import _backend, KERNEL_DIR

volume = _backend.reconstruct_gpu_opt(
    proj, angles,               # numpy arrays: proj [num_projs,H,W] float32, angles [num_projs] float64
    voxelSize, pixelSize, SDD, SOD,
    Nxz, Ny, W, H, num_projs,
    kernel_dir=KERNEL_DIR,
    epochs=100, subsets=1,      # subsets>1 = OSEM
)
# volume: numpy float32 array, shape (Nxz, Nxz, Ny)
```

Four entry points:

| Function | Description |
|---|---|
| `reconstruct_cpu(...)` | OpenMP CPU reference |
| `reconstruct_gpu_buf(..., kernel_dir=...)` | Manual buffer gather, no texture cache |
| `reconstruct_gpu_img(..., kernel_dir=...)` | Hardware texture sampler |
| `reconstruct_gpu_opt(..., kernel_dir=..., subsets=...)` | Hardware sampler + tuned kernels; only mode supporting OSEM (`subsets>1`) |

All four return the reconstructed volume as a numpy `float32` array;
none touch HDF5 — read input and write output in Python (see `run.py`
for a full example).

Reference call site:

```bash
python3 run.py --mode gpu-opt --epochs 100 --subsets 1
python3 run.py --mode cpu     --epochs 20
```

## Notes

- **First import is slow (~30-60s)** — that's the JIT compile.
  Subsequent imports reuse ninja's cached build and are fast. If a
  build ever looks corrupted or stuck:
  ```bash
  rm -rf ~/.cache/torch_extensions/ct_recon-*
  ```
- **VRAM**: `reconstruct_gpu_opt` checks the requested `subsets`
  against the device's available memory before running, and raises a
  Python exception instead of letting an out-of-memory OpenCL
  allocation kill the process. 512³ volumes need `subsets=1` on a 4GB
  card.
- **Ctrl-C** works during a reconstruction — the GIL is released for
  the duration of the C call.

## Verification

Compare against the CLI binary on identical input — both run the exact
same compiled C/OpenCL code:

```bash
make run-gpu-opt
python3 run.py --mode gpu-opt --out output_gpu_opt_py.hdf5
python3 python/validate.py   # or a direct h5py diff of the two Volume datasets
```

## Files

```
backend.py            — pybind11/torch JIT loader, four reconstruct_* entry points
run.py                — reference call site: HDF5 in/out around the pybind backend
src/
  ct_recon_bindings.cpp — pybind11 module, wraps utils.c/ct_cpu.c/ct_gpu.c
  main.c, utils.c/h, ct_cpu.c/h, ct_gpu.c/h — CLI + shared C/OpenCL implementation
kernels/               — OpenCL kernels (fp/bp, buffer and image variants)
python/                — validation scripts, course reference script
pybindextension/       — course's original minimal pybind11 example, unmodified
```
