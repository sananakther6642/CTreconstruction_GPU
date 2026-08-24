# CT Volume Reconstruction — GPU Lab Project

Cone-beam CT reconstruction using iterative MLEM. CPU (OpenMP) and GPU
(OpenCL) implementations, 256³ and 512³ datasets. All GPU modes
validated against CPU at the float32 noise floor.

Full investigation history (root causes, sweeps, negative results):
`sessions/2026-08-25-readme-cleanup-investigation-log.md` (local only).
Project methodology narrative: `sessions/2026-08-25-project-methodology-start-to-end.md`.

## Performance

### pool15-01 (AMD Hawaii PRO), EPOCHS=10

| Mode | 256³ time/epoch | 512³ time/epoch | Speedup vs CPU |
|---|---|---|---|
| `cpu` | 2.60 s | 22.9-23.3 s | 1× |
| `gpu-buf` | 0.44 s | 4.87-10.19 s (run-to-run variance, AMD-specific) | 5.9× / 2.3-4.8× |
| `gpu-img` | 0.095 s | 0.870-0.876 s | 27.4× / 26.6× |
| `gpu-opt` | 0.093 s | 0.873-0.879 s | 28.0× / 26.5× |

Intel i7-5820K (12 threads) · AMD Hawaii PRO (2560 shaders, 2.56 TFLOPS).

### kale (NVIDIA GTX 680), EPOCHS=100

| Mode | 256³ time/epoch | 256³ total | 512³ time/epoch | 512³ total | Speedup vs CPU |
|---|---|---|---|---|---|
| `cpu` | ~4.20s | 423.86s | ~34s | 3415.58s | 1× |
| `gpu-buf` | 0.855-0.860s | 86.46s | 7.13-7.20s | 722.59s | 4.9× / 4.7× |
| `gpu-img` | 0.142-0.148s | 14.67s | 1.226-1.256s | 126.74s | 28.9× / 27.0× |
| `gpu-opt` | 0.138-0.143s | 14.29s | 1.226-1.245s | 125.77s | 29.7× / 27.2× |

Intel Xeon E5-2620 0 (24 threads) · NVIDIA GTX 680 (Kepler, no
`cl_khr_fp16` — `--half` unavailable).

MSE vs CPU: 256³ `1.148e-10` (`gpu-buf`) / `1.128e-07` (`gpu-img`/`gpu-opt`).
512³ `9.534e-11` (`gpu-buf`) / `1.232e-09` (`gpu-img`/`gpu-opt`). No NaN/inf.

OSEM `--subsets 5`, 256³, 100 epochs: 57.17s total.

`gpu-buf` variance on pool15-01 (75-102s for the same config) does not
reproduce on kale (flat to the ms). Root cause: AMD-driver memory-
placement demotion of the volume buffer, not thermal throttling. Full
writeup in the session log.

## Validation (100 epochs, both datasets, all four modes)

### 256³
```
Mode       min      max     mean   nan  inf  MSE vs CPU         MSE vs Python
python   0.0002   0.1225   0.0144    0    0  (python ref)      -
cpu      0.0000   1.7303   0.0067    0    0  (reference)        MSE=1.215e-03
gpu-buf  0.0000   1.7307   0.0067    0    0  MSE=6.436e-10  max=0.0511
gpu-img  0.0000   1.8741   0.0067    0    0  MSE=1.949e-07  max=1.1490
gpu-opt  0.0000   1.8741   0.0067    0    0  MSE=1.949e-07  max=1.1490
```

### 512³
```
Mode       min      max     mean   nan  inf  MSE vs CPU
cpu      0.0000   1.0055   0.0330    0    0  (reference)
gpu-buf  0.0000   1.0054   0.0330    0    0  MSE=9.087e-11  max=0.0156
gpu-img  0.0000   1.0054   0.0330    0    0  MSE=6.849e-10  max=0.0169
gpu-opt  0.0000   1.0054   0.0330    0    0  MSE=6.849e-10  max=0.0169
```

Correctness fixes that got here (full detail in session log):
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
  at the Makefile default; `clFinish` batching — no win on kale.

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
validate.py                 — MSE + outlier diagnostics (256/512)
validate_ops.py             — per-operator fp/bp vs Python reference
run_python_reference.py     — full MLEM loop in Python
diag_fp.py                  — fp_cpu vs Python fp_func comparison
diag_voxel.py, diag_voxel2.py — one-off debugging scripts, kept for reference
Topic_2_CTreconstruction.py — original course-provided Python reference
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
python3 validate.py

# 512³
make run-cpu-512     EPOCHS=100
make run-gpu-buf-512 EPOCHS=100
make run-gpu-img-512 EPOCHS=100
make run-gpu-opt-512 EPOCHS=100
python3 validate.py 512

# component tests (CPU only, isolate fp/bp correctness)
make run-op-fp       # dumps fp_cpu.hdf5 (256³)
make run-op-bp       # dumps bp_cpu.hdf5 (256³)
make run-op-fp-512   # dumps fp_cpu_512.hdf5 (512³, use SAMPLES512=64 for a quick check)
make run-op-bp-512   # dumps bp_cpu_512.hdf5 (512³)
python3 validate_ops.py fp --data /lgrp/edu-2026-1-gpulab/proj_256_75.hdf5 --dump fp_cpu.hdf5
python3 validate_ops.py bp --data /lgrp/edu-2026-1-gpulab/proj_256_75.hdf5 --dump bp_cpu.hdf5

# Python reference (--epochs must match the C run being compared against)
make run-python     EPOCHS=10   # 256³, slow
make run-python-512 EPOCHS=10   # 512³, slower; sample_ratio mismatch, see session log
```

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
- 256³ only. Confirmed on kale (GTX 680, 4037MiB VRAM): plain `gpu-opt`
  at 512³ (S=1) already uses ~3.27GB, leaving ~760MB headroom — not
  enough for even one 512MB per-subset normalizer buffer at S=2.
- Regression tests: `--subsets` unset vs baseline, MSE unchanged;
  `--subsets 1` vs unset, byte-identical.

**Result** (256³, 100 epochs, log-likelihood vs wall-clock):

| Time | S=1 | S=3 | S=5 | S=15 | S=25 |
|---|---|---|---|---|---|
| 10s | −482,720 | −481,297 | **−481,109** | −480,543 | −480,577 |
| 15s | −481,109 | −480,391 | −480,304 | **−479,652** | −479,285 |
| 20s | −480,652 (plateaued) | −480,013 | −479,942 | −479,305 | **−478,917** |
| 30s | −480,652 (plateaued) | −479,701 | −479,653 | −479,042 | **−478,651** |
| 36s | −480,652 (plateaued) | −479,610 | −479,580 | −478,964 | **−478,571** |

S=1 plateaus by ~20s (100-epoch ceiling). Every OSEM config keeps
improving through 36s, never plateaus. S=25 is worst before ~10s
(sub-iteration overhead) but wins from ~15s on. Report the
time-matched curve, not a single "best S."

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
| Skip float32 staging copy | `ct_gpu.c` | direct `d_vol`→image copy in float32 mode |
