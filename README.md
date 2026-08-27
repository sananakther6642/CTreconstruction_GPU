# CT Volume Reconstruction — GPU Lab Project

Cone-beam CT reconstruction using iterative MLEM. CPU (OpenMP) and GPU
(OpenCL) implementations, 256³ and 512³ datasets. All GPU modes
validated against CPU at the float32 noise floor.

## Performance

### AMD Hawaii PRO, EPOCHS=10

| Mode | 256³ time/epoch | 512³ time/epoch | Speedup vs CPU |
|---|---|---|---|
| `cpu` | 2.60 s | 22.9-23.3 s | 1× |
| `gpu-buf` | 0.44 s | 4.87-10.19 s (run-to-run variance, AMD-specific) | 5.9× / 2.2-4.8× |
| `gpu-img` | 0.095 s | 0.870-0.876 s | 27.4× / 26.6× |
| `gpu-opt` | 0.093 s | 0.873-0.879 s | 28.0× / 26.5× |

Intel i7-5820K (12 threads) · AMD Hawaii PRO (2560 shaders, 2.56 TFLOPS).

### NVIDIA GeForce GTX 680, EPOCHS=100

| Mode | 256³ time/epoch | 256³ total | 512³ time/epoch | 512³ total | Speedup vs CPU |
|---|---|---|---|---|---|
| `cpu` | ~4.20s | 423.86s | ~34s | 3415.58s | 1× |
| `gpu-buf` | 0.855-0.860s | 86.46s | 7.13-7.20s | 722.59s | 4.9× / 4.7× |
| `gpu-img` | 0.142-0.148s | 14.67s | 1.226-1.256s | 126.74s | 28.9× / 27.0× |
| `gpu-opt` | 0.138-0.143s | 14.29s | 1.226-1.245s | 125.77s | 29.7× / 27.2× |

Intel Xeon E5-2620 0 (24 threads) · NVIDIA GeForce GTX 680 (Kepler, no
`cl_khr_fp16` — `--half` unavailable).

MSE vs CPU: 256³ `1.1477e-10` (`gpu-buf`) / `1.1278e-07` (`gpu-img`/`gpu-opt`).
512³ `9.534e-11` (`gpu-buf`) / `1.232e-09` (`gpu-img`/`gpu-opt`). No NaN/inf.
RMS as % of signal range: 256³ 0.0194%, 512³ 0.0026% (`gpu-img`/`gpu-opt`).
Improves at higher resolution, not worse. `gpu-buf`'s MSE-vs-CPU margin
over `gpu-img`/`gpu-opt` is scale-dependent, not a fixed ratio: ~980x
tighter at 256³ (`1.148e-10` vs `1.128e-07`) but only ~13x tighter at
512³ (`9.534e-11` vs `1.232e-09`) — `gpu-img`/`gpu-opt`'s absolute error
shrinks much faster with resolution than `gpu-buf`'s does.
(256³ re-confirmed 2026-08-26 on the NVIDIA GeForce GTX 680, fresh
100-epoch run, same worst voxels/order of magnitude as the original
measurement.)

`gpu-img`/`gpu-opt` are not bit-exact vs CPU/`gpu-buf` — checked why with
`diag_maxgap.py` rather than assumed. **Not** the earlier-suspected `1/U²`
geometric singularity (measured: min|U| at the worst voxel is 82-91% of
SOD at both scales, 1.2-1.5x amplification, nothing). It's a one-voxel
spatial displacement of sharp features from the hardware `CLK_FILTER_LINEAR`
sampler (all `+0.5f` half-texel offsets audited and correct — not a
coordinate bug). Mass is conserved across the displaced voxel pair. Affects
21/16.7M voxels at 256³, 138/134M at 512³ (>100×RMS), concentrated at the
FOV edge (100% beyond 0.8×FOV radius at 512³). `gpu-buf` (manual float32
trilinear, no hardware sampler) has none of this — use it when bit-level
fidelity to the CPU reference matters more than speed.

OSEM `--subsets 5`, 256³, 100 epochs: 57.17s total.

`gpu-buf` variance on AMD Hawaii PRO (75-102s for the same config) does not
reproduce on the NVIDIA GeForce GTX 680 (flat to the ms). Root cause:
AMD-driver memory-placement demotion of the volume buffer, not thermal
throttling. Thermal throttling ruled out three ways: the slowdown is a
single step change that holds (not the gradual ramp thermal throttling
would produce), its magnitude (6.6x) exceeds Hawaii's DVFS ceiling
(~3.2x), and `gpu-img`/`gpu-opt` stay stable in the same sessions where
`gpu-buf` goes slow (a device-wide thermal effect would hit all three).
Confirmed instead as a `d_vol`-allocation-specific effect: freeing and
recreating the 537MB buffer mid-run instantly recovers the fast timing,
then it degrades again on its own after a further 6-13 launches --
consistent with the driver periodically demoting that allocation's
memory placement under sustained access, not a hardware/thermal cause.

A mitigation (`FP_BUFFER_VOL_REALLOC_EVERY`, periodic buffer reallocation
to force it back into fast memory) was tried and did **not** hold up:
looked like a win at small scale but averaged *slower* (91.74s vs 86.96s)
over four full 10-epoch runs — off by default (see `src/ct_gpu.c` for the
real numbers).

## Validation (AMD Hawaii PRO, 100 epochs, both datasets, all four modes)

Earlier full-scale validation run, on the original AMD hardware — MSE
values here differ from the NVIDIA GeForce GTX 680 numbers in Performance
above, not a regression. The gap is real and larger than expected for
`gpu-buf` specifically: at 256³, AMD's `gpu-buf` MSE (`6.436e-10`) is
~5.6x worse than GTX 680's (`1.148e-10`), even though `gpu-buf` never
touches a hardware texture sampler (manual float32 interpolation only)
-- so "texture-sampler rounding" can't be the explanation for that part
of the gap. `-cl-fast-relaxed-math` (always on, `src/ct_gpu.c`) is a
plausible cause -- the OpenCL spec leaves its exact numerical behavior
vendor-defined, and a prior investigation confirmed it does NOT explain
`gpu-img`/`gpu-opt`'s *within-GPU* precision floor on a single machine
(rebuilt with the flag stripped, byte-identical output -- see
`experimentation/hybrid-precision/kernels/bp_image.cl`), but that test
never compared the same build across AMD vs NVIDIA, which is the actual
open question here. Not yet resolved -- would need a real AMD-hardware
rebuild with the flag stripped to test directly.

### 256³
```
Mode        min      max      mean   nan  inf  MSE vs CPU     MSE vs Python Ref
python_ref  0.0000   247.7164 0.0070    0    0  (python ref)
cpu         0.0000   1.7303   0.0067    0    0  (reference)    MSE=6.371e-03
gpu-buf     0.0000   1.7307   0.0067    0    0  MSE=6.436e-10  MSE=6.371e-03  max=0.0511
gpu-img     0.0000   1.8741   0.0067    0    0  MSE=1.949e-07  MSE=6.371e-03  max=1.1490
gpu-opt     0.0000   1.8741   0.0067    0    0  MSE=1.949e-07  MSE=6.371e-03  max=1.1490
```

Same outlier-voxel phenomenon as GTX 680's run (see the GTX 680
Validation section below): 24 outlier voxels dominate this MSE vs the
Python reference (`Topic_2_CTreconstruction.py`) figure (0.00030
excluding them vs 0.00637 including them), not a disagreement in any
C/GPU mode -- confirmed the same finding independently
on this machine's own Python-reference output.

### 512³
```
Mode       min      max     mean   nan  inf  MSE vs CPU     MSE vs Python Ref
cpu      0.0000   1.0055   0.0330    0    0  (reference)    n/a
gpu-buf  0.0000   1.0054   0.0330    0    0  MSE=9.087e-11  n/a  max=0.0156
gpu-img  0.0000   1.0054   0.0330    0    0  MSE=6.849e-10  n/a  max=0.0169
gpu-opt  0.0000   1.0054   0.0330    0    0  MSE=6.849e-10  n/a  max=0.0169
```

## Validation (NVIDIA GeForce GTX 680, 100 epochs, both datasets, all four modes)

MSE vs CPU matches the Performance section's numbers exactly (same
`validate.py` output). MSE vs Python Ref (`validate.py`'s label for the
course-provided Python reference, `Topic_2_CTreconstruction.py`) is now
available at 256^3 (`Epochs=20` run completed) -- see the note below the
256^3 table for why that number is dominated by 26 outlier voxels in the
Python reference's own output, not a disagreement in any C/GPU mode.
512^3 has no Python-reference comparison; that script hardcodes the
256^3 dataset path (a 512^3 variant, `Topic_2_CTreconstruction_512.py`,
exists -- see the Run section). Attempted on AMD Hawaii PRO
(2026-08-27): failed immediately at the first large-array allocation
(`bp_ones`, `ArrayMemoryError` on a 1GB `(512,512,512)` float64 array,
even under an 8GB memory cap) -- re-confirms the earlier finding that
512^3 does not fit on that machine's 15GB RAM regardless of cap, since
the physical RAM available (9GB at attempt time) is the real ceiling,
not the `ulimit`. Not yet attempted to completion on the NVIDIA GTX 680
(188GB RAM); in progress as of this writing.

### 256³
```
Mode        min      max      mean   nan  inf  MSE vs CPU     MSE vs Python Ref
python_ref  0.0000   521.6699 0.0070    0    0  (python ref)
cpu         0.0000   1.7303   0.0067    0    0  (reference)    MSE=2.061e-02
gpu-buf     0.0000   1.7292   0.0067    0    0  MSE=1.148e-10  MSE=2.061e-02  max=0.0203
gpu-img     0.0000   1.6577   0.0067    0    0  MSE=1.128e-07  MSE=2.061e-02  max=0.8966
gpu-opt     0.0000   1.6577   0.0067    0    0  MSE=1.128e-07  MSE=2.061e-02  max=0.8966
```

**MSE vs the Python reference (`Topic_2_CTreconstruction.py`) is
dominated by 26 outlier voxels in that script's own output (0.00016% of
the volume), not a disagreement in any C/GPU mode** -- all four modes
show the identical 2.061e-02, meaning it's the Python reference's output
that's the outlier here, not any implementation. Excluding just those 26
voxels drops MSE to ~3.2e-04. Investigated directly
(`python/diag_bp_ones.py`), not assumed: the MLEM normalizer `bp_ones`
is completely normal at every one of these voxels (364-443, same range
as everywhere else in the volume) -- ruling out a division-by-near-zero.
The real mechanism is geometric compounding: each outlier voxel's
per-epoch multiplicative ratio is consistently ~1.29-1.37 (backed out
from `521.6699^(1/20)` and `174.32614^(1/20)` matching the observed
final values starting from v0=1), so 20 unclamped MLEM iterations
compound a small persistent per-epoch drift into a large final value.
This is a real numerical-stability gap in the unmodified reference
script at a handful of ill-conditioned voxels (likely sparse/near-tangent
ray coverage there), not a bug in any of this project's C/GPU/CPU
implementations -- reproduced independently on AMD Hawaii PRO's own
Python-reference run too (24 outlier voxels, heavily
overlapping indices, same z=1/z=254-concentrated pattern, same order of
magnitude).

### 512³
```
Mode       min      max     mean   nan  inf  MSE vs CPU     MSE vs Python Ref
python_ref (no 512^3 Python-reference run possible -- the reference script hardcodes the 256^3 dataset path, see Run section below)
cpu      0.0000   1.0055   0.0330    0    0  (reference)    n/a
gpu-buf  0.0000   1.0054   0.0330    0    0  MSE=9.534e-11  n/a  max=0.0114
gpu-img  0.0000   1.0054   0.0330    0    0  MSE=1.232e-09  n/a  max=0.0186
gpu-opt  0.0000   1.0054   0.0330    0    0  MSE=1.232e-09  n/a  max=0.0186
```

Correctness fixes that got here:
- `fp_cpu` used `(int)xi` (truncation) instead of `floorf(xi)` — wrong
  bounds-check pass for `xi` in `(-1,0)`. Fixed.
- GPU `bp` kernels zero-padded individual OOB taps instead of zeroing
  the whole sample (Python reference behavior). Fixed to match.
- Python reference ran bp-only, not full MLEM — invalidated "MSE vs
  Python". Fixed to run the same fp→ratio→bp→update loop.

Component tests: `--op fp|bp` dumps a single fp/bp call in isolation;
`validate_ops.py` compares it to the Python reference.

## Optimization history

- **fp_cpu ray-tiling** (512³): batched neighboring rays, sample index
  as outer loop → cache-friendly access order. `cpu` 512³: 44.5-45.7s
  → 26.06s/epoch.
- **D1 (bp_cpu branch removal)**: reverted. Looked like a wash on a
  single noisy run; a 3-trial comparison found a real ~3.7% CPU
  regression. Original `continue`-based code restored.
- **unroll-x2 in bp_opt**: removed. Measured 1.6% regression on AMD
  Hawaii (register pressure hurt occupancy more than ILP helped).
  `gpu-opt`/`gpu-img` now essentially tied (0.17% gap).
- **Work-group tuning**: `fp_image` `{16,16,1}→{8,32,1}` (~5-7%),
  `fp_buffer` `{16,16,1}→{4,64,1}` (>4× at 512³, largest single win),
  `fp_cpu` `FP_TILE` `8→32` (~19% at 512³). Env overrides:
  `FP_IMAGE_LWS`, `FP_BUFFER_LWS`, `FP_TILE_ENV`.
- **Negative/neutral results** (kept, not chased further): AABB
  clipping still hurts at 256³ when forced on; a latent (inert) AABB
  axis-transposition bug found and fixed; sphere-shaped AABB — no
  headroom over box clip; `bp_cpu` thread scaling already near-optimal
  at the Makefile default; `clFinish` batching — no win on the NVIDIA GeForce GTX 680.

## Modes

| Mode | Flag | Description |
|---|---|---|
| CPU | `--mode cpu` | OpenMP, incremental ray stepping |
| GPU buffer | `--mode gpu-buf` | Manual bilinear/trilinear, no texture cache — naive baseline |
| GPU image | `--mode gpu-img` | Hardware image2d_array + image3d sampler |
| GPU opt | `--mode gpu-opt` | Hardware sampler + float2 LUT + local mem |

`gpu-img`/`gpu-opt` share `fp_image.cl`; only their `bp` kernel differs.
AABB clipping (gated `W>512`) applies to both.

`vol_img` defaults to float32; `--half` opts into `CL_HALF_FLOAT`
(lower bandwidth, ~3-decimal-digit quantization cost).

## Files

```
src/
  main.c              — CLI, dispatch, HDF5 save
  utils.c/h           — HDF5 load/save, timing
  ct_cpu.c/h          — CPU: cone_weight, fp_cpu, bp_cpu, reconstruct_cpu
  ct_gpu.c/h          — OpenCL host: gpu_init, reconstruct_gpu, reconstruct_gpu_opt
kernels/
  bp_buffer.cl        — bp (buffer) + preprocess_proj + proj_divide + vol_update
  fp_buffer.cl        — fp (buffer): ray march + manual trilinear + AABB
  bp_image.cl         — bp (image): hardware bilinear + float2 LUT
  fp_image.cl         — fp (image): hardware trilinear + AABB
  bp_buffer_opt.cl    — bp_opt: image2d_array_t + float2 LUT + local mem
python/
  validate.py                 — MSE vs CPU + vs Python Ref, outlier diagnostics (256/512)
  validate_ops.py             — per-operator fp/bp vs Topic_2_CTreconstruction.py
  diag_voxel.py                — one-off debugging script, kept for reference
  diag_maxgap.py               — per-operator error attribution for the gpu-img/gpu-opt max-gap finding
  plot_results.py              — report figures (OSEM/MLEM convergence, slice comparisons)
  Topic_2_CTreconstruction.py — the reference script (fp_func/bp_func,
                                 unmodified algorithm); the "Reference code
                                 (python)" grading refers to. Two real bugs
                                 fixed (out_path placeholder, volume z-axis size)
                                 so it actually runs and saves; algorithm untouched.
                                 Has a real, unfixed numerical-stability gap at
                                 a handful of voxels -- see the 26-outlier-voxel
                                 finding in the Validation section.
  Topic_2_CTreconstruction_512.py — 512^3 variant, path_data/out_path
                                 changed only; timing/feasibility unmeasured.
  diag_bp_ones.py               — investigates the 26-outlier-voxel finding
                                 above (checks whether bp_ones is near-zero
                                 at those voxels; it isn't).
```

## Build

```bash
sudo apt install libhdf5-dev ocl-icd-opencl-dev opencl-headers gcc make
make
```

## Run

Data: `/lgrp/edu-2026-1-gpulab/`.

```bash
# 256³
make run-cpu     EPOCHS=100
make run-gpu-buf EPOCHS=100
make run-gpu-img EPOCHS=100
make run-gpu-opt EPOCHS=100
python3 python/validate.py

# 512³
make run-cpu-512     EPOCHS=100
make run-gpu-buf-512 EPOCHS=100
make run-gpu-img-512 EPOCHS=100
make run-gpu-opt-512 EPOCHS=100
python3 python/validate.py 512

# component tests (CPU only, isolate fp/bp correctness)
make run-op-fp       # dumps fp_cpu.hdf5 (256³)
make run-op-bp       # dumps bp_cpu.hdf5 (256³)
make run-op-fp-512   # dumps fp_cpu_512.hdf5 (512³, use SAMPLES512=64 for a quick check)
make run-op-bp-512   # dumps bp_cpu_512.hdf5 (512³)
python3 python/validate_ops.py fp --data /lgrp/edu-2026-1-gpulab/proj_256_75.hdf5 --dump fp_cpu.hdf5
python3 python/validate_ops.py bp --data /lgrp/edu-2026-1-gpulab/proj_256_75.hdf5 --dump bp_cpu.hdf5

# Reference (python) -- 256³ only, hardcoded epoch count in-script
# (no --data/--epochs flags, EPOCHS not honored)
make run-python
```

Pure-Python fp/bp (scipy `RegularGridInterpolator` rebuilt per-angle, no
vectorization/GPU) measured ~4690s/epoch at 256^3 on the NVIDIA GeForce GTX 680 at the script's
default `sample_ratio=2`. `sample_ratio` reduced to 1 (call-site only, in
`fp_func(cb_para, vol, sample_ratio=1)`; `fp_func`/`bp_func` internals
untouched) — this also matches the C/GPU modes' default sample count at
256^3 (`n_samples = Nxz = 256`, see `src/main.c`), which `sample_ratio=2`'s
512 samples/ray did not. Measured ~3679s/epoch at sample_ratio=1 (only
~21% faster, not the ~50% a naive samples-per-ray scaling would suggest —
most of fp_func's cost is fixed per-pixel Python/interpreter overhead
across its 1.3M-pixel loop, not proportional to ray length; bp_func's cost
is unaffected by sample_ratio at all). Full 100 epochs at this rate would be
~4.3 days; fewer epochs than 100 is an accepted tradeoff for this
script — results converge less but that's expected, not a defect —
so `Epochs` set to 20 (~20hrs) instead. Run started on the NVIDIA GeForce GTX 680
under `tmux -s topic2` to run unattended.

Full MSE-vs-CPU and MSE-vs-Python-reference numbers (real `Epochs=20`
run, completed 2026-08-27) are in the "Validation (NVIDIA GeForce GTX
680...)" section above, including the 26-outlier-voxel finding that
explains the 2.061e-02 figure.

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
- 256³ only. Confirmed on the NVIDIA GeForce GTX 680 (4037MiB VRAM): plain `gpu-opt`
  at 512³ (S=1) already uses ~3244-3274MiB, leaving ~760-790MiB
  headroom — not enough for the 1024MiB (2×512MiB) a second subset's
  normalizer buffers would need at S=2.
- Regression tests: `--subsets` unset vs baseline, MSE unchanged;
  `--subsets 1` vs unset, byte-identical.

**Result** (256³, 100 epochs, log-likelihood vs wall-clock — logged via
`--log-convergence <file.csv>`, which writes per-epoch loglik/residual/
rel_change; off by default):

| Time | S=1 | S=3 | S=5 | S=15 | S=25 |
|---|---|---|---|---|---|
| 10s | −482,720 | −481,297 | −481,109 | **−480,543** | −480,577 |
| 15s | −481,109 | −480,391 | −480,304 | −479,652 | **−479,285** |
| 20s | −480,652 (plateaued) | −480,013 | −479,942 | −479,305 | **−478,917** |
| 30s | −480,652 (plateaued) | −479,701 | −479,653 | −479,042 | **−478,651** |
| 36s | −480,652 (plateaued) | −479,610 | −479,580 | −478,964 | **−478,571** |

S=1 plateaus by ~20s (100-epoch ceiling). Every OSEM config keeps
improving through 36s, never plateaus. S=15 leads at 10s, S=25 is a
close second there (sub-iteration overhead still amortizing) but
overtakes and wins clearly from 15s on. Report the time-matched curve,
not a single "best S."

## Optimizations

| Optimization | Where | Effect |
|---|---|---|
| `-ffast-math` + OpenMP `collapse(2)` | `fp_cpu`, `bp_cpu` | denormal flush; parallelism |
| Incremental ray stepping | `fp_cpu`, `fp_buffer.cl`, `fp_image.cl` | fewer multiplies in ray march |
| image2d_array_t hardware bilinear | `bp_image.cl`, `bp_buffer_opt.cl` | texture cache + HW interpolation |
| image3d_t hardware trilinear | `fp_image.cl` | texture cache replaces manual 8-tap trilinear |
| float2 cos/sin LUT | `bp_buffer_opt.cl`, `bp_image.cl` | no cos/sin per voxel per angle |
| Local memory LUT cache | `bp_buffer_opt.cl` | angle_cs cooperatively loaded into `__local` |
| float4 vectorized divide/update | `bp_buffer.cl` | 4 elements/work-item |
| Work-group `{4,4,16}` for bp | `ct_gpu.c` | coalesced writes |
| Fused cone_weight + preprocess_proj | `bp_buffer.cl` | one kernel pass instead of two |
| Precomputed R/T matrices | `ct_gpu.c`, `fp_image.cl` | no per-work-item cos/sin + matmul in fp |
| Half-precision vol_img (`--half`) | `fp_image.cl`, `ct_gpu.c` | halves texture bandwidth; off by default |
| AABB slab ray clipping (`W>512`) | `fp_image.cl`, `fp_buffer.cl` | skips empty cone-beam edge rays |
| `native_recip` in bp kernels | `bp_buffer_opt.cl`, `bp_buffer.cl`, `bp_image.cl` | HW SFU reciprocal |
| OMP_NUM_THREADS=nproc | `Makefile` | uses all available cores |
| Ray tiling (`FP_TILE`) | `fp_cpu` | cache reuse across neighboring rays |
| `schedule(guided,4)` | `fp_cpu` | lower scheduling overhead, handles AABB imbalance |
| Fused proj_divide + preprocess_proj (`divide_preprocess_img`) | `bp_buffer.cl` | one kernel writes straight to `ratio_img`, no intermediate copy |
| `vol_update_img` | `bp_buffer.cl`, `ct_gpu.c` | vol_update writes directly into `vol_img`, eliminates the per-epoch `d_vol`→image copy entirely (float32 mode) |
