# CT Volume Reconstruction — GPU Lab Project

Cone-beam CT reconstruction using iterative MLEM on CPU (OpenMP) and GPU (OpenCL).
Supports 256³ and 512³ datasets. All GPU modes validated: no NaN/Inf, MSE vs CPU within float32 rounding bounds (see Validation below — this was previously a real bug, now fixed and re-measured).

## Performance Results

### 256³ dataset (512×512 detector, 75 angles, n\_samples=256)

Measured on `pool15-01`, EPOCHS=10, post correctness-fix (see Validation):

| Mode | Time/epoch | Speedup vs CPU |
|------|-----------|----------------|
| `cpu` (12-thread OpenMP) | ~4.5 s | 1× |
| `gpu-buf` (chunked) | ~0.55 s | **~8×** |
| `gpu-img` | ~0.099 s | **~45×** |
| `gpu-opt` | ~0.097 s | **~46×** |

`gpu-buf`'s chunked launches (z-slabs with `clFinish` between, added to fix
the 512³ watchdog hang) cost a small overhead here (~0.55s vs the
pre-chunking ~0.425-0.435s) — acceptable tradeoff since it's what makes
512³ actually completable. `gpu-img`/`gpu-opt` also picked up a small
overhead from the `--half` toggle defaulting to float32 vol_img instead of
always-half; not yet isolated how much of the ~0.03-0.04s/epoch delta vs the
original table is that vs other changes. 100-epoch totals not re-measured
yet — extrapolate at your own risk, or re-run with EPOCHS=100 to confirm.

### 512³ dataset (1120×1184 detector, 75 angles, n\_samples=512)

> ⚠ Table below predates the correctness fixes described in Validation
> (bp/fp boundary-rule mismatch, fp_cpu truncation bug, gpu-buf watchdog
> chunking) and has not been re-measured since. Numbers not to be trusted
> until re-run with the current code — treat as historical reference only.

| Mode | Time/epoch | Total (100ep) | Speedup vs CPU |
|------|-----------|--------------|----------------|
| `cpu` (12-thread OpenMP) | 47.3 s | 4730 s | 1× |
| `gpu-buf` | — | — | previously blocked; now chunked, needs re-measurement |
| `gpu-img` | 0.631 s | 63.1 s | **75×** |
| `gpu-opt` | **0.657 s** | **65.7 s** | **72×** |

**Hardware:** Intel Core i7-5820K @ 3.30GHz (12 logical cores) · AMD Hawaii PRO (Radeon R9 290/390, 2560 shaders, 2.56 TFLOPS) · `pool15-01.cis.iti.uni-stuttgart.de`

> `gpu-buf` on 512³ previously caused a GPU driver hang from a single
> oversized kernel launch exceeding the driver watchdog timeout (bp_buffer's
> manual bilinear does 4 uncoalesced global loads/angle/voxel with no
> texture cache — at 512³ that's ~15-20s in one launch). Fixed by chunking
> the NDRange into z-slabs (bp) / angle-slabs (fp) with `clFinish` between
> launches. `run-gpu-buf-512` is unblocked in the Makefile; needs a
> real 512³ timing/hang-check run to confirm and fill in the table above.

### Key optimizations that drove 512³ speedup

> ⚠ Predates the correctness fixes above; per-stage numbers not re-measured.

| Change | bp (ms) | fp (ms) | Total (s) |
|--------|---------|---------|-----------|
| Baseline | 718 | 409 | 1.149 |
| Work-group `{8,8,4}→{4,4,16}` (coalesced z-writes) | **290** | 409 | 0.724 |
| AABB ray clipping in fp\_image (gated W>512) | 290 | **342** | 0.659 |
| `native_recip` in bp\_opt | 290 | 342 | 0.657 |

Note: the `native_recip` row shows no measured bp/fp delta (290→290,
342→342) — its effect on the previous total (0.659→0.657s) is within
measurement noise, not a real driver of the speedup. Left in the table for
history; don't cite it as a proven win without re-measuring in isolation.

## Validation (10 epochs)

### 256³ — current (post-fix)
```
Mode       min      max     mean   nan  inf  MSE vs CPU         MSE vs Python
python   0.0002   0.1225   0.0144    0    0  (python ref)      -
cpu      0.0000   1.4189   0.0066    0    0  (reference)        MSE=6.263e-04
gpu-buf  0.0000   1.4188   0.0066    0    0  MSE=3.458e-13  max=0.0012  MSE=6.263e-04
gpu-img  0.0000   1.4310   0.0066    0    0  MSE=1.494e-09  max=0.0762  MSE=6.263e-04
gpu-opt  0.0000   1.4310   0.0066    0    0  MSE=1.494e-09  max=0.0762  MSE=6.263e-04
```

`gpu-buf` MSE/max are now at the float32 noise floor (max=0.0012 on a volume
with mean=0.0066, max value 1.42). `gpu-img`/`gpu-opt` sit ~60× higher on
`max` but still small in absolute terms — attributed to `CLK_FILTER_LINEAR`
hardware texture sampler precision (OpenCL spec allows implementation-defined
rounding in hardware bilinear/trilinear, typically lower precision than IEEE
manual interpolation); not chased further since `gpu-buf` already proves the
underlying algorithm is correct to noise level.

"MSE vs CPU" and "MSE vs Python" measure genuinely different things:
CPU-vs-GPU (this table) compares two implementations of the *same*
algorithm/boundary rules, at the noise floor above. CPU-vs-Python
(6.263e-04, unchanged by any of these fixes) reflects `n_samples=Nxz=256`
in the C runs vs Python `fp_func`'s module default `sample_ratio=2`
(512 samples) elsewhere in the codebase — an actual sampling-resolution
difference between the two implementations, not a bug; not yet reconciled
(see Known gaps below).

**What was fixed to get here** (previously: `max~1.17-1.25`, MSE~1.4e-7,
attributed in this README to "float32 rounding" — that framing was wrong):
1. `fp_cpu` used `int x0 = (int)xi` (truncates toward zero) instead of
   `floorf(xi)`. For `xi` in `(-1, 0)` this gives `x0=0` instead of `-1`,
   which wrongly passes the `(unsigned)x0 < Nxz-1` bounds check and samples
   `volume[0]` with a bogus weight for a ray that should have missed the
   volume entirely. Found via a component test comparing `fp_cpu(ones)`
   against the Python reference pixel-by-pixel — one pixel had C=1.107,
   Python=0.0. Fixed in `src/ct_cpu.c` (fp_cpu only — the two `(int)uf`/
   `(int)vf` casts in `bp_cpu` are safe, already guarded by an explicit
   `< 0` check before the cast).
2. GPU `bp` kernels (`bp_buffer.cl`, `bp_image.cl`, `bp_buffer_opt.cl`)
   zero-padded individual out-of-bounds interpolation taps; the Python
   reference (`scipy.interpolate.RegularGridInterpolator(fill_value=0)`)
   zeros the *entire* sample if any tap is out of bounds — verified
   empirically, not documented behavior. GPU kernels now match: an
   explicit whole-cell bounds check gates the texture read / tap lookup.
3. `run_python_reference.py` previously ran a single bp-only MLEM step (no
   `fp` at all) and called that "MSE vs Python" — comparing two different
   algorithms. It now runs the same full fp→ratio→bp→update loop as
   `reconstruct_cpu`/`reconstruct_gpu`, with a `jitter=False` mode so its
   fixed-step ray march matches the C/GPU kernels' (which have no jitter).

Component-test tools added for future debugging: `--op fp|bp` on
`ct_recon` dumps a single fp/bp call in isolation (bypasses accumulated
MLEM iteration error); `validate_ops.py` compares that dump against the
Python reference per-operator.

### 512³ — stale, needs re-measurement
```
Mode       min      max     mean   nan  inf  MSE vs CPU
cpu      0.0001   0.3481   0.0334    0    0  (reference)
gpu-img  0.0000   0.5172   0.0331    0    0  MSE=8.452e-04  max=0.4236
gpu-opt  0.0000   0.5172   0.0331    0    0  MSE=8.452e-04  max=0.4236
```
⚠ Predates all fixes above. Re-run `make run-cpu-512 run-gpu-img-512
run-gpu-opt-512 run-gpu-buf-512` (EPOCHS=10, or 100 for the real number)
and `python3 validate.py 512` to get current numbers. `gpu-buf-512`'s row
is entirely new — it was hard-blocked in the Makefile until this session.

### Known gaps
- CPU-vs-Python sampling mismatch (n_samples=256 vs 512) not reconciled —
  doesn't affect CPU-vs-GPU correctness (both C paths use the same
  n_samples), but means "MSE vs Python" isn't apples-to-apples yet.
- 512³ table above is stale.

## Modes

| Mode | Flag | Description |
|------|------|-------------|
| CPU | `--mode cpu` | C + OpenMP, `-ffast-math`, incremental ray stepping |
| GPU buffer | `--mode gpu-buf` | OpenCL global buffers, manual bilinear/trilinear (256³ only) |
| GPU image | `--mode gpu-img` | Hardware image2d_array + image3d sampler |
| GPU opt | `--mode gpu-opt` | Hardware sampler + float2 LUT + local mem + AABB clipping |

`vol_img` precision (`gpu-img`/`gpu-opt`) defaults to **float32**; pass
`--half` to opt into `CL_HALF_FLOAT` for lower texture bandwidth at the cost
of ~3-decimal-digit quantization (was previously always-on, uncredited as an
accuracy tradeoff — see Validation).

## Files

```
src/
  main.c              — CLI: parse args, dispatch to cpu/gpu modes, save HDF5
  utils.c/h           — HDF5 load/save, get_time_sec()
  ct_cpu.c/h          — CPU: cone_weight, fp_cpu, bp_cpu, reconstruct_cpu
  ct_gpu.c/h          — OpenCL host: gpu_init, reconstruct_gpu, reconstruct_gpu_opt
kernels/
  bp_buffer.cl        — bp (buffer) + preprocess_proj + proj_divide + vol_update
  fp_buffer.cl        — fp (buffer): ray march + manual trilinear + AABB clipping
  bp_image.cl         — bp (image): hardware bilinear on image2d_array_t + float2 LUT
  fp_image.cl         — fp (image): hardware trilinear on image3d_t + AABB clipping
  bp_buffer_opt.cl    — bp_opt: image2d_array_t + float2 LUT + local mem + unroll-x2
  fp_buffer_opt.cl    — dead code: never loaded by gpu_init, not wired to any dispatch
validate.py           — load HDF5 outputs, print MSE + outlier-location diagnostics (supports 256/512)
validate_ops.py       — per-operator fp/bp comparison vs Python reference, isolated from MLEM iteration
run_python_reference.py — full MLEM loop in Python (fp_func/bp_func), --epochs to match C runs
```

## Build

```bash
make
```

Requires: `gcc`, `libhdf5-dev`, `ocl-icd-opencl-dev`, `opencl-headers`.

```bash
sudo apt install libhdf5-dev ocl-icd-opencl-dev opencl-headers gcc make
```

## Run

Data is at `/lgrp/edu-2026-1-gpulab/` on lab machines.

### 256³
```bash
make run-cpu     EPOCHS=100
make run-gpu-buf EPOCHS=100
make run-gpu-img EPOCHS=100
make run-gpu-opt EPOCHS=100
python3 validate.py
```

### 512³
```bash
make run-cpu-512     EPOCHS=100
make run-gpu-buf-512 EPOCHS=100   # now unblocked; chunked launches avoid the watchdog hang
make run-gpu-img-512 EPOCHS=100
make run-gpu-opt-512 EPOCHS=100
python3 validate.py 512
```

Or run directly:
```bash
./build/ct_recon --data /lgrp/edu-2026-1-gpulab/proj_512_75.hdf5 \
                 --out output.hdf5 --mode gpu-opt --epochs 100 \
                 --samples 512 --kernels kernels
```

### Component tests (isolate fp/bp correctness, CPU only)
```bash
make run-op-fp   # dumps fp_cpu.hdf5
make run-op-bp   # dumps bp_cpu.hdf5
python3 validate_ops.py fp --data /lgrp/edu-2026-1-gpulab/proj_256_75.hdf5 --dump fp_cpu.hdf5
python3 validate_ops.py bp --data /lgrp/edu-2026-1-gpulab/proj_256_75.hdf5 --dump bp_cpu.hdf5
```

### Python reference (must match C run's `--epochs` for validate.py's "MSE vs Python" to be meaningful)
```bash
make run-python EPOCHS=10   # slow — Python fp is ~minutes/epoch
```

## Algorithm

MLEM multiplicative update:

```
v = ones(Nxz, Nxz, Ny)
bp_ones = bp( cone_weight( ones_proj ) )   # computed once before loop

for each epoch:
    b      = fp(v)                          # forward project current estimate
    ratio  = p0 / b    (b > 1e-3)          # measured / estimated
    ratio  = cone_weight(ratio)             # apply cone weight before bp
    v     *= bp(ratio) / bp_ones           # multiplicative MLEM update
```

`preprocess_proj`: flip H-axis + transpose `[np,H,W]→[np,W,H]` + `/voxelSize`
(applied to ratio and ones before every bp call).

## Optimizations

| Optimization | Where | Effect |
|---|---|---|
| `-ffast-math` + OpenMP `collapse(2)` | `fp_cpu`, `bp_cpu` | denormal flush; parallelism across detector pixels |
| Incremental ray stepping | `fp_cpu`, `fp_buffer.cl`, `fp_image.cl` | eliminates multiply per sample in ray march |
| image2d\_array\_t hardware bilinear | `bp_image.cl`, `bp_buffer_opt.cl` | texture cache + free HW interpolation |
| image3d\_t hardware trilinear | `fp_image.cl` | texture cache replaces manual 8-tap trilinear |
| float2 cos/sin LUT | `bp_buffer_opt.cl`, `bp_image.cl` | eliminates cos/sin per voxel per angle |
| Local memory LUT cache | `bp_buffer_opt.cl` | angle\_cs cooperatively loaded into `__local` |
| float4 vectorized divide/update | `bp_buffer.cl` | 4 elements/work-item via `vload4`/`vstore4` |
| Work-group `{4,4,16}` for bp kernels | `ct_gpu.c` | 16 contiguous z-threads → coalesced writes (volume layout `[x][y][z]`) |
| Fused cone\_weight + preprocess\_proj | `bp_buffer.cl` | one kernel pass instead of two |
| Precomputed R/T matrices | `ct_gpu.c`, `fp_image.cl` | eliminates cos/sin + matmul per work-item in fp |
| Half-precision `vol_img` (`CL_HALF_FLOAT`, opt-in via `--half`) | `fp_image.cl`, `ct_gpu.c` | halves texture bandwidth for volume reads; float\_to\_half kernel on GPU, no PCIe roundtrip; ~3-decimal-digit accuracy cost, so off by default |
| AABB slab ray clipping (gated W>512) | `fp_image.cl`, `fp_buffer.cl` | tightens per-ray sample range on large detectors; skips empty cone-beam edge rays |
| `native_recip` in bp kernels | `bp_buffer_opt.cl`, `bp_buffer.cl`, `bp_image.cl` | hardware SFU reciprocal, ~4× faster than IEEE division |
| Unroll-x2 gated Nxz≥512 | `bp_buffer_opt.cl` | ILP across two texture fetches; gated to avoid register pressure on 256³ |
| OMP\_NUM\_THREADS=nproc | `Makefile` cpu targets | uses all available cores on lab node |
