# CT Volume Reconstruction — GPU Lab Project

Cone-beam CT reconstruction using iterative MLEM on CPU (OpenMP) and GPU (OpenCL).
Supports 256³ and 512³ datasets. All GPU modes validated: no NaN/Inf, MSE vs CPU within float32 rounding bounds (see Validation below — this was previously a real bug, now fixed and re-measured).

## Performance Results

### 256³ dataset (512×512 detector, 75 angles, n\_samples=256)

Measured on `pool15-01`, **EPOCHS=100**, current code (post all fixes):

| Mode | Time/epoch | Total (100ep) | Speedup vs CPU |
|------|-----------|--------------|----------------|
| `cpu` (12-thread OpenMP) | 2.73 s | 273.1 s | 1× |
| `gpu-buf` (chunked) | 0.57 s | 57.4 s | **4.8×** |
| `gpu-img` | 0.101 s | 10.1 s | **27.2×** |
| `gpu-opt` | **0.098 s** | **9.8 s** | **27.9×** |

### 512³ dataset (1120×1184 detector, 75 angles, n\_samples=512)

Measured on `pool15-01`, **EPOCHS=100**, current code (post all fixes):

| Mode | Time/epoch | Total (100ep) | Speedup vs CPU |
|------|-----------|--------------|----------------|
| `cpu` (12-thread OpenMP) | 26.06 s | 2605.5 s | 1× |
| `gpu-buf` (chunked) | 56.93 s | 5693.2 s | **0.46× (slower than CPU)** |
| `gpu-img` | 0.930 s | 93.0 s | **28.0×** |
| `gpu-opt` | ~0.925 s (10-epoch, see note) | ~92.5 s (extrapolated) | **~28.2×** |

**Hardware:** Intel Core i7-5820K @ 3.30GHz (12 logical cores) · AMD Hawaii PRO (Radeon R9 290/390, 2560 shaders, 2.56 TFLOPS) · `pool15-01.cis.iti.uni-stuttgart.de`

> **`gpu-opt` row is a 10-epoch measurement, not 100**: after removing the
> unroll-x2 path (see "gpu-opt vs gpu-img" below), two separate 10-epoch
> runs confirmed **0.922-0.930s/epoch** consistently — now genuinely
> faster than `gpu-img`'s 0.930s instead of marginally slower, as
> intended. 100-epoch confirmation not run yet; re-run
> `make run-gpu-opt-512 EPOCHS=100` for the final number (extrapolation
> above assumes the same per-epoch cost holds at 100, which prior modes'
> 10-vs-100-epoch numbers in this table have shown to be a reasonable but
> not perfect assumption).

> **`gpu-buf` on 512³ is slower than CPU** — genuinely, not a bug. It
> previously caused a driver hang from one oversized kernel launch
> exceeding the watchdog timeout; fixed by chunking the NDRange into
> z-slabs (bp) / angle-slabs (fp) with `clFinish` between launches, so it
> now completes reliably (confirmed over a full 100-epoch, ~95-minute run).
> But per-slab timing diagnostics show every `fp_buffer` angle-slab
> consistently takes 2.5-8s — `fp_buffer.cl`'s manual 8-tap trilinear
> gather (no texture cache, 8 uncoalesced global reads per sample) is real,
> honest memory-bandwidth-bound work this hardware can't do fast at 512³
> scale, and the CPU's much smaller memory footprint plus the ray-tiling
> speedup (below) now beats it outright. Not chased further: `gpu-img`/
> `gpu-opt` exist specifically to avoid this cost via hardware texture
> sampling, and they deliver the real ~28× speedup.

> At 256³, `gpu-buf` is a legitimate (if modest) 4.8× win — the same
> kernel, just at a scale where the memory traffic fits within what the
> hardware can handle reasonably. The naive/optimized contrast is really
> about the transition between these two scales.

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

## Validation (100 epochs)

Full-scale run: both datasets, all four modes, matching epoch counts
throughout — the epoch-mismatch caveat from earlier 10-epoch checks no
longer applies anywhere below.

### 256³
```
Mode       min      max     mean   nan  inf  MSE vs CPU         MSE vs Python
python   0.0002   0.1225   0.0144    0    0  (python ref)      -
cpu      0.0000   1.7303   0.0067    0    0  (reference)        MSE=1.215e-03
gpu-buf  0.0000   1.7307   0.0067    0    0  MSE=6.436e-10  max=0.0511  MSE=1.215e-03
gpu-img  0.0000   1.8741   0.0067    0    0  MSE=1.949e-07  max=1.1490  MSE=1.215e-03
gpu-opt  0.0000   1.8741   0.0067    0    0  MSE=1.949e-07  max=1.1490  MSE=1.215e-03
```

### 512³
```
Mode       min      max     mean   nan  inf  MSE vs CPU
cpu      0.0000   1.0055   0.0330    0    0  (reference)
gpu-buf  0.0000   1.0054   0.0330    0    0  MSE=9.087e-11  max=0.0156
gpu-img  0.0000   1.0054   0.0330    0    0  MSE=6.849e-10  max=0.0169
gpu-opt  0.0000   1.0054   0.0330    0    0  MSE=6.849e-10  max=0.0169
```

All three GPU modes agree with CPU at essentially the float32 noise floor
at both scales — confirms the correctness fixes below hold up over full
100-epoch convergence, not just the shorter 10-epoch checks used while
debugging. `max` is naturally larger at 100 epochs than at 10 (compounding
iteration), but MSE stays tiny throughout.

**What was fixed to get here** (earlier this project, `max` was ~1.17-1.25
at only 10 epochs, attributed in this README to "float32 rounding" — that
framing was wrong):
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

### CPU 512³ speedup: fp_cpu ray-tiling rewrite
`fp_cpu` dominates the CPU epoch time at 512³ (was 35.2-36.2s of a
44.5-45.7s epoch — 78%). The 8-tap trilinear gather it does per ray sample
is memory-latency-bound: each gather spans up to ~2MB due to the
`Nxz*Ny=262144` and `Ny=512` float strides in the volume layout, so
sequentially marching one ray to completion before starting the next gave
the CPU cache nothing to reuse.

Fix: batch `FP_TILE=8` neighboring detector rows together, and iterate the
sample index `s` as the *outer* loop across the tile instead of per-ray.
Neighboring rays for the same angle start near each other and diverge only
slightly, so at a given `s` their volume addresses cluster — batching
means the memory subsystem sees those nearby reads close together in time
instead of scattered across a full ray march per pixel. Each ray still
computes its own AABB-derived `s_start`/`s_end` and only accumulates within
its own range, so the actual math and reduction order per ray is unchanged
— this is purely a memory-access-order change, verified via `--op fp`
component test to still agree with the Python reference (`MSE=8.34e-08` at
64 samples, no regression from the pre-tiling `8.57e-08` at full samples).

Result at 100 epochs: `cpu` 512³ **26.06s/epoch** (2605.5s total), down
from a pre-tiling baseline of 44.5-45.7s/epoch — confirmed stable across
the full 100-epoch run, not just the earlier 10-epoch spot check.
`bp_cpu` itself was never the bottleneck (a separate analytical-range
optimization there had negligible effect since bp's cost is dominated by
the same kind of gather, just at a fraction of fp's time budget) — the win
is entirely from fp_cpu.

### gpu-opt vs gpu-img: unroll-x2 was measured harmful, removed
`gpu-opt` (`bp_opt` in `bp_buffer_opt.cl`) layers a float2 cos/sin LUT,
cooperative local-memory caching, and (previously) an unroll-x2 loop on
top of the same hardware texture sampler `gpu-img` uses. It should
therefore always be at least as fast as `gpu-img` — but the 100-epoch
512³ run showed `gpu-opt` at **0.945s/epoch, slightly slower** than
`gpu-img`'s 0.930s. Investigated properly instead of accepting a "close
enough" 1.6% gap:

1. Ruled out redundant `floor()` calls in `bp_opt`'s bounds check (it
   called `floor(u)`/`floor(v)` twice each vs `bp_image.cl`'s once) —
   fixed this regardless (unconditionally correct cleanup), re-measured:
   **no change** (0.934-0.940s), so this wasn't the cause.
2. Tested the unroll-x2 path in isolation by disabling its gate
   (`Nxz>=512` → an always-false condition) to force the scalar-only loop
   at 512³: **0.923-0.927s/epoch — faster than `gpu-img`**, confirming
   unroll-x2 was the actual cause of the regression, not masking it.

Unroll-x2's theory was that overlapping two texture fetches would hide
latency via instruction-level parallelism. On this hardware (AMD Hawaii,
GCN 1.1) it did the opposite: the doubled live register set (two
`float2`/`float4`/etc. sets in flight instead of one) apparently costs
more in work-group occupancy than the ILP saves in latency-hiding — the
code's original comment gated this "for large volumes" as an assumption,
never actually measured on this GPU. Removed permanently; `bp_opt` is now
unconditionally scalar. Confirmed with two independent 10-epoch runs after
removal — **0.923-0.927s** and **0.922-0.930s** — consistently faster
than `gpu-img`'s 0.930s, as the LUT/local-mem work was always meant to
deliver. 100-epoch confirmation still needed (see Performance Results note).

### Known gaps
- CPU-vs-Python sampling mismatch (n_samples=256 vs Python `fp_func`'s
  module default `sample_ratio=2`, i.e. 512 samples, at 256³) not
  reconciled — doesn't affect CPU-vs-GPU correctness (both C paths use the
  same n_samples), but means "MSE vs Python" isn't apples-to-apples yet.
  `MSE vs Python` at 512³ is also not shown above: 512³'s C output and the
  Python reference (which only ran at 256³) have different shapes.
- `gpu-opt` 512³ Performance Results row needs a 100-epoch re-run to
  replace the pre-unroll-removal number (see above).

## Modes

| Mode | Flag | Description |
|------|------|-------------|
| CPU | `--mode cpu` | C + OpenMP, `-ffast-math`, incremental ray stepping |
| GPU buffer | `--mode gpu-buf` | OpenCL global buffers, manual bilinear/trilinear — naive baseline; slow at 512³ (no texture cache), chunked to avoid the driver watchdog |
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
  bp_buffer_opt.cl    — bp_opt: image2d_array_t + float2 LUT + local mem
  fp_buffer_opt.cl    — dead code: never loaded by gpu_init, not wired to any dispatch
validate.py           — load HDF5 outputs, print MSE + outlier-location diagnostics (supports 256/512)
validate_ops.py       — per-operator fp/bp comparison vs Python reference, isolated from MLEM iteration
run_python_reference.py — full MLEM loop in Python (fp_func/bp_func), --epochs to match C runs
diag_fp.py             — fp_cpu vs Python fp_func comparison, parameterized for 256^3/512^3 (--data/--dump/--samples)
diag_voxel.py, diag_voxel2.py — one-off scripts from the boundary-rule/truncation bug hunt; kept for reference, not part of the regular workflow
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
Both print their own elapsed time (`fp_cpu time: X.XXX s`). 512³ variants
exist since fp/bp cost differs a lot by dataset size.
```bash
make run-op-fp       # dumps fp_cpu.hdf5 (256^3)
make run-op-bp       # dumps bp_cpu.hdf5 (256^3)
make run-op-fp-512   # dumps fp_cpu_512.hdf5 (512^3, SAMPLES512 samples — slow, use SAMPLES512=64 for a quick check)
make run-op-bp-512   # dumps bp_cpu_512.hdf5 (512^3)
python3 validate_ops.py fp --data /lgrp/edu-2026-1-gpulab/proj_256_75.hdf5 --dump fp_cpu.hdf5
python3 validate_ops.py bp --data /lgrp/edu-2026-1-gpulab/proj_256_75.hdf5 --dump bp_cpu.hdf5
python3 diag_fp.py --data /lgrp/edu-2026-1-gpulab/proj_512_75.hdf5 --dump fp_cpu_512.hdf5 --samples 512
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
| ~~Unroll-x2 gated Nxz≥512~~ (removed) | `bp_buffer_opt.cl` | tried, measured harmful on this GPU — see "gpu-opt vs gpu-img" below |
| OMP\_NUM\_THREADS=nproc | `Makefile` cpu targets | uses all available cores on lab node |
| Ray tiling (`FP_TILE=8`, sample-index-outer loop) | `fp_cpu` | groups neighboring rays' 8-tap gathers close in time for cache reuse; 2.1× on fp_cpu at 512³ |
| `schedule(guided,4)` | `fp_cpu` | lower scheduling overhead than `dynamic` at 512³'s larger iteration count (75×1120=84000 work items), still handles AABB-clipped load imbalance |
