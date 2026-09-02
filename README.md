# CT Volume Reconstruction - GPU Lab Project

Cone-beam CT reconstruction using iterative MLEM. CPU (OpenMP) and GPU
(OpenCL) implementations, 256³ and 512³ datasets. Packaged per the
course's required pattern: a pybind11/torch Python interface
(`backend.py`, `run.py`) wraps the same compiled C/OpenCL sources the
CLI binary uses - no duplicated logic between the two entry points.

**Correctness summary:**
- All 3 GPU modes (`gpu-buf`, `gpu-img`, `gpu-opt`) agree with the CPU
  reference at the float32 noise floor (MSE 1e-7 to 1e-10 vs CPU, both
  datasets, both machines - see Performance/Validation below).
- MSE vs the course's Python reference (`Topic_2_CTreconstruction.py`)
  was initially higher (`2.061e-02`/`6.371e-03` depending on hardware) -
  confirmed a numerical-stability bug in that script's unclamped MLEM
  update, not in this project's code. 26 voxels on one machine, 24 on
  the other (<0.0002% of the volume) diverged over 20 iterations.
  Reported to course staff, approved to fix. Fix: clamp `v0` to `[0,5]`
  after each update in `Topic_2_CTreconstruction.py` and its 512³
  variant; `fp_func`/`bp_func` themselves untouched.
- Re-verified with a real 20-epoch run on AMD Hawaii PRO: MSE vs Python
  Ref `6.371e-03` → `3.014e-04` (~21× better), all 4 modes agree
  identically, `python_ref` max exactly `5.0000` (the clamp bound).
  Independently reproduced on NVIDIA GTX 680 (`2.06e-02` → `1.749e-04`).
- One qualification: the fix **bounds** the unstable voxels rather than
  removing them. Counted directly: 56 voxels ran away unclamped
  (reaching 521.67); 47 (Hawaii) / 54 (GTX 680) now sit pinned at the
  clamp bound with it. MSE improves because capping 521.67→5.0 cuts a
  voxel's squared error by ~10⁴, not because those voxels became
  correct - containment, not elimination. See the Validation section
  for full numbers.

## pybind11 / torch Python interface

Python-callable interface to the CT reconstruction backend (`src/`),
built with `torch.utils.cpp_extension.load` per the course's required
pattern (see `pybindextension/` for the original minimal example this
follows).

### Requirements

- Linux (the C sources `#include <hdf5.h>` and `<omp.h>` unconditionally;
  neither is set up for macOS here - build and run on pool15 / kale).
- Python packages: `torch`, `numpy`, `h5py`.
- `ninja` (torch's JIT build requires it; not always pulled in
  automatically - `pip install ninja` if `from backend import _backend`
  fails with "Ninja is required").
- HDF5 dev headers/libs and an OpenCL ICD (see Build below).

### Usage

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
none of them touch HDF5 - read input and write output in Python (see
`run.py`).

Demo / reference call site:

```bash
python3 run.py --mode gpu-opt --epochs 100 --subsets 1
```

### Notes

- **First import is slow (~30-60s)** - that's the JIT compile.
  Subsequent imports reuse ninja's cached build (tracked by file
  mtimes) and are fast. If a build ever looks corrupted or stuck,
  delete the cache and retry:
  ```bash
  rm -rf ~/.cache/torch_extensions/ct_recon-*
  ```
- **`-march=native` and `-ffast-math`** match the Makefile's CLI build.
  The JIT build directory is namespaced per-host (see `backend.py`), so
  a native-arch binary is never shared across machines - safe to use.
  Without `-march=native`, `reconstruct_cpu` measured ~1.8× slower on
  kale (7.2s/epoch vs the CLI's 3.9s/epoch, same OMP env on both
  sides); with it, timings match. Output agrees with the CLI binary at
  the float32 noise floor (see Verification below).
- **CPU thread pinning**: `backend.py` sets `OMP_NUM_THREADS` /
  `OMP_PROC_BIND=close` / `OMP_PLACES=cores` at import time (matching
  the Makefile's `run-cpu` target) unless already set in the
  environment. Without pinning, `reconstruct_cpu`'s per-epoch time was
  observed climbing steadily on kale (5.5s → 14s+ over ~25 epochs) as
  OpenMP threads drifted across cores; with pinning it stays flat.
- **Env vars must be set before `import torch`, not just before the C
  call**: `backend.py` sets those three vars via
  `os.environ.setdefault(...)` above the `torch.utils.cpp_extension`
  import. This ordering matters, not just the values - torch's own C++
  init reads/caches OpenMP configuration once at import time, and
  GOMP's runtime is shared in-process with torch's. Setting the vars
  *after* `import torch` (even via `setdefault`, even moments later)
  left `reconstruct_cpu` a flat, reproducible ~13% slower than the CLI
  on both `fp` and `bp` despite identical values ending up in
  `os.environ` either way.
- **Flush-to-zero / denormals-are-zero**: `-ffast-math` sets FTZ/DAZ via
  `crtfastmath.o`, which is only linked into a *main executable* - the
  CLI's `main.c` gets it for free, but this extension's main executable
  is `python3`, built without `-ffast-math`, so compiling the `.so`
  with the flag alone did not set MXCSR here. Fixed by setting FTZ/DAZ
  explicitly in `ct_recon_bindings.cpp`'s `PYBIND11_MODULE` init
  (`_MM_SET_FLUSH_ZERO_MODE` / `_MM_SET_DENORMALS_ZERO_MODE`); OpenMP
  worker threads inherit MXCSR from the thread that set it, so one call
  at import time is sufficient. Confirmed on kale: `fp_cpu` holds flat
  for 40+ epochs, output bit-identical to the CLI (`MSE: 0.0`).
- **VRAM**: `reconstruct_gpu_opt` checks the requested `subsets` against
  the device's available memory before running, and raises a Python
  exception (with the computed requirement) instead of letting an
  out-of-memory OpenCL allocation kill the process. 512³ volumes need
  `subsets=1` on a 4GB card; higher subset counts need more VRAM than
  is available (`~(3+subsets) × 537MB + 4 × 79MB` at 512³).
- **Ctrl-C** works during a reconstruction - the GIL is released for the
  duration of the C call.

### Verification

```bash
python3 run.py --mode gpu-opt --out output_gpu_opt_py.hdf5
python3 python/validate.py   # MSE vs CPU + vs Python reference, both datasets
```

The pybind interface produces bit-identical output to a separately
built binary linking the same sources, at both resolutions, on Hawaii
(verified, all four modes) - expected, since the binding JIT-compiles
the same `src/ct_gpu.c` the rest of the project's results are measured
from.

## Performance

### AMD Hawaii PRO, EPOCHS=100

| Mode | 256³ time/epoch | 512³ time/epoch | Speedup 256³ | Speedup 512³ |
|---|---|---|---|---|
| `cpu` | 2.635 s | 22.377 s | 1× | 1× |
| `gpu-buf` | 0.576 s | 45.523 s† | 4.6× | 0.5×† |
| `gpu-img` | 0.086 s | 1.083 s | 30.6× | 20.7× |
| `gpu-opt` | **0.080 s** | **0.740 s** | **32.9×** | **30.2×** |

†`gpu-buf` at 512³ is the one figure here that is not a stable
measurement - see the DVFS variance note below. Every other cell is
reproducible. Intel i7-5820K (12 threads) · AMD Hawaii PRO (2560
shaders, 2.56 TFLOPS).

### NVIDIA GeForce GTX 680, EPOCHS=100

| Mode | 256³ time/epoch | 256³ total | 512³ time/epoch | 512³ total | Speedup |
|---|---|---|---|---|---|
| `cpu` | ≈4.20s | 423.86s | ≈34s | 3415.58s | 1× |
| `gpu-buf` | 0.855-0.860s | 86.46s | 7.13-7.20s | 722.59s | 4.9× / 4.7× |
| `gpu-img` | 0.142-0.148s | 14.67s | 1.226-1.256s | 126.74s | 28.9× / 27.0× |
| `gpu-opt` | **0.138-0.143s** | **14.29s** | **1.226-1.245s** | **125.77s** | **29.7× / 27.2×** |

Intel Xeon E5-2620 0 (24 threads) · NVIDIA GeForce GTX 680 (Kepler, no
`cl_khr_fp16` - `--half` unavailable).

- MSE vs CPU: 256³ `1.1477e-10` (`gpu-buf`) / `1.1278e-07`
  (`gpu-img`/`gpu-opt`). 512³ `9.534e-11` (`gpu-buf`) / `1.232e-09`
  (`gpu-img`/`gpu-opt`). No NaN/inf.

### `FP_TEX_EXACT`: exact forward-projection interpolation

`gpu-img`/`gpu-opt` sample the volume through the hardware texture unit,
which bundles two separable things: a 3D-tiled cache (the source of
their speed) and a fixed-function interpolation stage whose blend
weights carry less than float32 precision (the source of their MSE gap
vs CPU). Setting `FP_TEX_EXACT=1` keeps the cache and replaces only the
blend - eight `CLK_FILTER_NEAREST` fetches plus a manual float32
trilinear, in forward projection specifically. fp produces the ratio
that drives every MLEM update, so its error re-enters the loop each
epoch, while bp's averages out across 75 angles. Off by default;
opt-in.

**MSE vs CPU, 100 epochs:**

| | NVIDIA GTX 680 default | NVIDIA GTX 680 exact | AMD Hawaii PRO default | AMD Hawaii PRO exact |
|---|---|---|---|---|
| 256³ `gpu-img`/`gpu-opt` | `1.128e-07` | `2.524e-09` | `1.949e-07` | `1.651e-09` |
| 512³ `gpu-img`/`gpu-opt` | `1.232e-09` | `5.528e-10` | `6.849e-10` | `4.359e-10` |
| 256³ `gpu-buf` | `1.148e-10` | n/a | `6.436e-10` | n/a |
| 512³ `gpu-buf` | `9.534e-11` | n/a | `5.310e-10` | n/a |

`gpu-buf` has no `fp_image`, so the flag does not apply to it. Every
cell is measured; nothing is estimated. Improvement factors: 44.7×
(GTX680 256³), 118× (Hawaii 256³), 2.2× (GTX680 512³), 1.6× (Hawaii
512³) - the fix transfers across vendors and converges them to a
similar corrected level, and the gain shrinks with resolution since
512³ starts closer to the float32 noise floor.

At 512³ on Hawaii the corrected texture path (`4.359e-10`) is more
accurate than `gpu-buf` (`5.310e-10`) on the same data, while remaining
~29-37× faster - the accuracy-vs-speed tradeoff motivating a separate
manual-interpolation mode does not hold at that resolution on that
hardware.

**Runtime cost (100 epochs, one build per platform, so ratios are
meaningful):**

| Platform | Scale | default → exact (`gpu-img`) | default → exact (`gpu-opt`) | ratio |
|---|---|---|---|---|
| GTX 680 | 256³ | 16.95s → 21.33s | 16.51s → 20.98s | 1.26–1.27× |
| Hawaii | 256³ | 8.60s → 11.22s | 8.01s → 10.61s | 1.30×/1.32× |
| Hawaii | 512³ | 108.27s → 156.27s | 74.00s → 123.14s | 1.44×/1.66× |

A 45× MSE reduction for ~1.3× runtime at 256³. Defaults are unchanged;
the flag costs nothing unless set. Full write-up, including four
approaches that were tried and did not work, in
`docs/precision-256-investigation.md`.

- OSEM on Hawaii, MSE vs CPU: `1.062e-04` (S=5, 256³), `4.050e-05`
  (S=2, 512³). OSEM converges through a different update path - S
  partial updates per epoch rather than one full update - so a larger
  divergence from plain MLEM is inherent to the method, not a
  precision defect.
- `gpu-buf`'s MSE-vs-CPU margin over `gpu-img`/`gpu-opt` is
  scale-dependent: ~980× tighter at 256³, ~13× tighter at 512³.
- `gpu-img`/`gpu-opt`'s remaining default-path gap vs CPU/`gpu-buf` is a
  one-voxel spatial displacement from the hardware `CLK_FILTER_LINEAR`
  sampler, not a `1/U²` singularity (ruled out directly). Mass is
  conserved across the displaced voxel pair. `FP_TEX_EXACT` closes this
  gap (see above); `gpu-buf` (manual float32 trilinear) never had it -
  use `gpu-buf` when bit-level CPU fidelity matters more than speed and
  `FP_TEX_EXACT` isn't set.
- **`gpu-buf` run-to-run variance on AMD Hawaii PRO** (75-102s over 10
  epochs at 512³, does not reproduce on GTX 680): root-caused to
  memory-clock (mclk) DVFS, confirmed by direct clock-state
  instrumentation - every slow slab sampled `mclk=150MHz`, every fast
  baseline `mclk=1500MHz`, no overlap, core clock and temperature both
  ruled out. Two mitigations were tried and both failed once measured
  at the 100-epoch scale actually reported (`FP_BUFFER_VOL_REALLOC_EVERY`:
  91.74s vs 86.96s baseline over 4 full runs; `FP_BUFFER_SKIP_SLAB_FINISH`:
  looked like a 29% win at 30 epochs, measured 40.8 s/epoch - worse than
  doing nothing - at 100). Both off by default. The sysfs knob that
  would pin the top clock state is not writable without root on this
  machine.

## Validation (AMD Hawaii PRO, 100 epochs, both datasets, all four modes)

### 256³
```
Mode        min      max      mean   nan  inf  MSE vs CPU     MSE vs Python Ref
python_ref  0.0000   5.0000   0.0069    0    0  (python ref)
cpu         0.0000   1.7303   0.0067    0    0  (reference)    MSE=3.014e-04
gpu-buf     0.0000   1.7307   0.0067    0    0  MSE=6.436e-10  MSE=3.014e-04  max=0.0511
gpu-img     0.0000   1.8741   0.0067    0    0  MSE=1.949e-07  MSE=3.016e-04  max=1.1490
gpu-opt     0.0000   1.8741   0.0067    0    0  MSE=1.949e-07  MSE=3.016e-04  max=1.1490
```
MSE vs Python Ref: `6.371e-03` (unfixed) → `3.014e-04` (~21× better)
after the `v0` clamp fix. `python_ref` max exactly `5.0000` (clamp
bound, confirms engaging). 47 voxels sit pinned at that bound - bounded,
not eliminated (see Correctness summary above). MSE vs CPU unchanged,
as expected - the fix only touches the Python reference script.

### 512³
```
Mode       min      max     mean   nan  inf  MSE vs CPU     MSE vs Python Ref
cpu      0.0000   1.0055   0.0330    0    0  (reference)
gpu-buf  0.0000   1.0054   0.0330    0    0  MSE=9.087e-11  max=0.0156
gpu-img  0.0000   1.0054   0.0330    0    0  MSE=6.849e-10  max=0.0169
gpu-opt  0.0000   1.0054   0.0330    0    0  MSE=6.849e-10  max=0.0169
```
No Python-reference output at 512³ - ran out of memory on both machines
(15GB hard limit on Hawaii; GTX 680 has more RAM but still ran out
partway).

## Validation (NVIDIA GeForce GTX 680)

### 256³, 100 epochs C/GPU, 20 epochs Python reference
```
Mode        min      max      mean   nan  inf  MSE vs CPU     MSE vs Python Ref
python_ref  0.0000   5.0000   0.0069    0    0  (python ref)
cpu         0.0000   1.7303   0.0067    0    0  (reference)    MSE=1.749e-04
gpu-buf     0.0000   1.7292   0.0067    0    0  MSE=1.148e-10  MSE=1.749e-04  max=0.0203
gpu-img     0.0000   1.6577   0.0067    0    0  MSE=1.128e-07  MSE=1.749e-04  max=0.8966
gpu-opt     0.0000   1.6577   0.0067    0    0  MSE=1.128e-07  MSE=1.749e-04  max=0.8966
```
MSE vs Python Ref: `2.061e-02` (unfixed) → `1.749e-04` after the clamp
fix - reproduces the Hawaii result on a second vendor. This row is
scored against a 20-epoch CPU run (matching the Python script's own
epoch count), so it's not directly comparable to the 100-epoch
MSE-vs-CPU numbers elsewhere in this README. 54 voxels sit pinned at
the clamp bound. Over 99% of voxels agree to within 0.0046; median
difference 4.94e-05.

### 512³
```
Mode       min      max     mean   nan  inf  MSE vs CPU     MSE vs Python Ref
cpu      0.0000   1.0055   0.0330    0    0  (reference)
gpu-buf  0.0000   1.0054   0.0330    0    0  MSE=9.534e-11  max=0.0114
gpu-img  0.0000   1.0054   0.0330    0    0  MSE=1.232e-09  max=0.0186
gpu-opt  0.0000   1.0054   0.0330    0    0  MSE=1.232e-09  max=0.0186
```
No Python-reference output at 512³ - same out-of-memory limit as above.

**Correctness fixes that got here:**
- `fp_cpu` used `(int)xi` (truncation) instead of `floorf(xi)` - wrong
  bounds-check pass for `xi` in `(-1,0)`. Fixed; dropped CPU/GPU MSE by
  more than four orders of magnitude.
- GPU `bp` kernels zero-padded individual out-of-bounds taps instead of
  zeroing the whole interpolation cell (the Python reference's actual
  behavior). Fixed to match.

Component tests: `--op fp|bp` dumps a single fp/bp call in isolation;
`python/validate_ops.py` compares it to the Python reference.

## Optimization history

- **fp_cpu ray-tiling** (512³): batched neighboring rays, sample index
  as outer loop → cache-friendly access order. `cpu` 512³: 44.5-45.7s
  → 26.06s/epoch. `FP_TILE` is resolution-aware in the shipped code
  (384 at 512³, 32 at 256³) after a later re-sweep found 32 no longer
  optimal at 512³ on the GTX 680.
- **AABB ray-clipping**: a real win on `fp_cpu` at 512³ (~26% overall,
  ~32% on `fp_cpu` alone), but ~6% *slower* on `gpu-buf` even with the
  gate on - that path's cost is dominated by uncoalesced memory access,
  which range-narrowing doesn't address, so `gpu-buf`'s default was
  flipped to unconditionally off rather than sharing the CPU path's
  `W>512` threshold. Forcing AABB on at 256³ produces a ~4% regression
  on `fp_cpu`/`fp_image` (setup cost of 6 slab divisions outweighs the
  savings) - the gate correctly stays at `W>512` there.
- **D1 (bp_cpu branch removal)**: reverted. Looked like a wash on a
  single noisy run; a 3-trial comparison found a real ~3.7% CPU
  regression. Original `continue`-based code restored.
- **unroll-x2 in bp_opt**: removed. Measured 1.6% regression on AMD
  Hawaii (register pressure hurt occupancy more than ILP helped).
  `gpu-opt`/`gpu-img` now essentially tied (0.17% gap).
- **Work-group tuning**: `fp_image` `{16,16,1}→{8,32,1}` (~5-7%),
  `fp_buffer` `{16,16,1}→{2,16,2}` on GTX680 (Hawaii's `{4,64,1}`
  optimum does not transfer - re-tuned per platform). Env overrides:
  `FP_IMAGE_LWS`, `FP_BUFFER_LWS`, `FP_TILE_ENV`.

## Modes

| Mode | pybind function | Description |
|---|---|---|
| CPU | `reconstruct_cpu` | OpenMP, incremental ray stepping |
| GPU buffer | `reconstruct_gpu_buf` | Manual bilinear/trilinear, no texture cache - naive baseline |
| GPU image | `reconstruct_gpu_img` | Hardware image2d_array + image3d sampler |
| GPU opt | `reconstruct_gpu_opt` | Hardware sampler + float2 LUT + local mem |

`gpu-img`/`gpu-opt` share `fp_image.cl`; only their `bp` kernel differs.
AABB clipping (gated `W>512`) applies to `fp_cpu`/`fp_image`; disabled
unconditionally on `gpu-buf` (see Optimization history).

`vol_img` defaults to float32; `--half` opts into `CL_HALF_FLOAT`
(lower bandwidth, ~3-decimal-digit quantization cost). `FP_TEX_EXACT=1`
opts into exact float32 forward-projection interpolation (see above).

## Files

```
backend.py            - pybind11/torch JIT loader, four reconstruct_* entry points
run.py                - reference call site: HDF5 in/out around the pybind backend
src/
  ct_recon_bindings.cpp - pybind11 module, wraps utils.c/ct_cpu.c/ct_gpu.c
  main.c               - CLI, dispatch, HDF5 save
  utils.c/h            - HDF5 load/save, timing
  ct_cpu.c/h           - CPU: cone_weight, fp_cpu, bp_cpu, reconstruct_cpu
  ct_gpu.c/h           - OpenCL host: gpu_init, reconstruct_gpu, reconstruct_gpu_opt
kernels/
  bp_buffer.cl        - bp (buffer) + preprocess_proj + proj_divide + vol_update
  fp_buffer.cl        - fp (buffer): ray march + manual trilinear + AABB
  bp_image.cl         - bp (image): hardware bilinear + float2 LUT
  fp_image.cl         - fp (image): hardware trilinear + AABB, FP_TEX_EXACT
  bp_buffer_opt.cl    - bp_opt: image2d_array_t + float2 LUT + local mem
python/
  validate.py                 - MSE vs CPU + vs Python Ref, outlier diagnostics (256/512)
  validate_ops.py             - per-operator fp/bp vs Topic_2_CTreconstruction.py
  plot_results.py             - report figures (OSEM/MLEM convergence, slice comparisons)
  plot_fp_tex_exact.py        - FP_TEX_EXACT precision-fix figure
  Topic_2_CTreconstruction.py - reference script (fp_func/bp_func). v0 clamped
                                 to [0,5] after each update (course-staff
                                 approved fix for a numerical-stability gap;
                                 see Correctness summary). fp_func/bp_func
                                 themselves untouched.
  Topic_2_CTreconstruction_512.py - 512³ variant, same v0 clamp; ran out of
                                 memory on both machines (see Validation).
pybindextension/       - course's original minimal pybind11 example, unmodified
```

## Build

No separate build step - `backend.py` JIT-compiles on first import
(see pybind section above):
```bash
sudo apt install libhdf5-dev ocl-icd-opencl-dev opencl-headers
```

## Run

Data: `/lgrp/edu-2026-1-gpulab/`.

```bash
python3 run.py --mode gpu-opt --epochs 100
python3 run.py --mode cpu --epochs 20 --subsets 1
```

- Pure-Python fp/bp (scipy `RegularGridInterpolator` rebuilt per-angle,
  no vectorization/GPU): ~4690s/epoch at 256³ on GTX 680, script
  default `sample_ratio=2`. A complete 20-epoch run on the GTX 680 took
  20h 03m; steady-state cost (excluding the one-off first-epoch setup)
  is 3483.5 s/epoch, against which `gpu-opt` at 0.141 s/epoch is
  ~24,700× faster.

## Algorithm

MLEM multiplicative update:

```
v = ones(Nxz, Nxz, Ny)
bp_ones = bp( cone_weight( ones_proj ) )   # once, before the loop

for each epoch:
    b      = fp(v)
    ratio  = p0 / b    (b > 1e-3)
    ratio  = cone_weight(ratio)
    v     *= bp(ratio) / bp_ones
```

`preprocess_proj`: flip H-axis + transpose `[np,H,W]→[np,W,H]` +
`/voxelSize` (applied to ratio and ones before every bp call).

### OSEM (`gpu-opt` only, `--subsets N`)

Splits the angle set into `N` subsets, one volume update per subset
instead of one per full angle pass. `--subsets 1` (default) is
byte-identical to plain MLEM.

```
bp_ones[s] = bp( cone_weight(ones_proj), subset s )   for s in 0..N   # once

for each epoch (= one pass over all N subsets):
    for each subset s:
        b      = fp(v)                          # full angle range
        ratio  = p0 / b   (b > 1e-3)
        ratio  = cone_weight(ratio)
        v     *= bp(ratio, subset s) / bp_ones[s]   # subset's angles only
```

- Angle ordering: subset `s` gets `{s, s+N, s+2N, ...}` (interleaved),
  subsets visited in a golden-ratio-derived coprime stride order.
  Permutation applied once at load time (`utils.c`), so each subset
  becomes a contiguous `(ip_start, ip_count)` launch range.
- 256³ only. Confirmed on the NVIDIA GeForce GTX 680 (4037MiB VRAM):
  plain `gpu-opt` at 512³ (S=1) already uses ~3244-3274MiB, leaving
  ~760-790MiB headroom - not enough for the 1024MiB (2×512MiB) a
  second subset's normalizer buffers would need at S=2.
- Regression tests: `--subsets` unset vs baseline, MSE unchanged;
  `--subsets 1` vs unset, byte-identical.

**Result** (256³, 100 epochs, log-likelihood vs wall-clock - logged via
`--log-convergence <file.csv>`, off by default):

| Time | S=1 | S=3 | S=5 | S=15 | S=25 |
|---|---|---|---|---|---|
| 10s | −482,720 | −481,297 | −481,109 | **−480,543** | −480,577 |
| 20s | −480,652 (plateaued) | −480,013 | −479,942 | −479,305 | **−478,917** |
| 36s | −480,652 (plateaued) | −479,610 | −479,580 | −478,964 | **−478,571** |

S=1 plateaus by ~20s (100-epoch ceiling). Every OSEM config keeps
improving through 36s. S=15 leads at 10s, but S=25 overtakes it and
wins clearly from ~15s on - report the time-matched curve, not a
single "best S."

## Optimizations

| Optimization | Where | Effect |
|---|---|---|
| `-ffast-math` + OpenMP `collapse(2)` | `fp_cpu`, `bp_cpu` | denormal flush; parallelism |
| Incremental ray stepping | `fp_cpu`, `fp_buffer.cl`, `fp_image.cl` | fewer multiplies in ray march |
| image2d_array_t hardware bilinear | `bp_image.cl`, `bp_buffer_opt.cl` | texture cache + HW interpolation |
| image3d_t hardware trilinear | `fp_image.cl` | texture cache replaces manual 8-tap trilinear |
| `FP_TEX_EXACT` exact blend | `fp_image.cl` | closes the hardware sampler's precision gap, opt-in |
| float2 cos/sin LUT | `bp_buffer_opt.cl`, `bp_image.cl` | no cos/sin per voxel per angle |
| Local memory LUT cache | `bp_buffer_opt.cl` | angle_cs cooperatively loaded into `__local` |
| float4 vectorized divide/update | `bp_buffer.cl` | 4 elements/work-item |
| Work-group `{4,4,16}` for bp | `ct_gpu.c` | coalesced writes |
| Fused divide/preprocess/image-write (`divide_preprocess_img`) | `bp_buffer.cl` | one kernel writes straight to the ratio image, no intermediate copy |
| `vol_update_img` | `bp_buffer.cl`, `ct_gpu.c` | vol_update writes directly into `vol_img`, eliminates the per-epoch `d_vol`→image copy |
| Precomputed R/T matrices | `ct_gpu.c`, `fp_image.cl` | no per-work-item cos/sin + matmul in fp |
| Half-precision vol_img (`--half`) | `fp_image.cl`, `ct_gpu.c` | halves texture bandwidth; off by default |
| AABB slab ray clipping (`W>512`, `fp_cpu`/`fp_image` only) | `fp_image.cl`, `src/ct_cpu.c` | skips empty cone-beam edge rays |
| `native_recip` in bp kernels | `bp_buffer_opt.cl`, `bp_buffer.cl`, `bp_image.cl` | HW SFU reciprocal |
| OMP_NUM_THREADS=nproc | `Makefile` | uses all available cores |
| Ray tiling (`FP_TILE`, resolution-aware: 384 at 512³, 32 at 256³) | `fp_cpu` | cache reuse across neighboring rays |
| `schedule(guided,4)` | `fp_cpu` | lower scheduling overhead, handles AABB imbalance |
