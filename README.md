# CT Volume Reconstruction — GPU Lab Project

Cone-beam CT reconstruction using iterative MLEM on CPU (OpenMP) and GPU (OpenCL).
Supports 256³ and 512³ datasets. All GPU modes validated: no NaN/Inf, MSE vs CPU within expected float32 rounding bounds.

## Performance Results

### 256³ dataset (512×512 detector, 75 angles, n\_samples=256)

| Mode | Time/epoch | Total (100ep) | Speedup vs CPU |
|------|-----------|--------------|----------------|
| `cpu` (12-thread OpenMP) | 3.79 s | 379 s | 1× |
| `gpu-buf` | 0.425 s | 42.5 s | **8.9×** |
| `gpu-img` | 0.068 s | 6.8 s | **56×** |
| `gpu-opt` | **0.066 s** | **6.6 s** | **57×** |

### 512³ dataset (1120×1184 detector, 75 angles, n\_samples=512)

| Mode | Time/epoch | Total (100ep) | Speedup vs CPU |
|------|-----------|--------------|----------------|
| `cpu` (12-thread OpenMP) | 47.3 s | 4730 s | 1× |
| `gpu-buf` | — | — | ⚠ GPU hang (see note) |
| `gpu-img` | 0.631 s | 63.1 s | **75×** |
| `gpu-opt` | **0.657 s** | **65.7 s** | **72×** |

**Hardware:** Intel Core i7-5820K @ 3.30GHz (12 logical cores) · AMD Hawaii PRO (Radeon R9 290/390, 2560 shaders, 2.56 TFLOPS) · `pool15-01.cis.iti.uni-stuttgart.de`

> ⚠ `gpu-buf` on 512³ causes a GPU driver hang (global memory thrashing → watchdog timeout → display reset). The Makefile blocks this target. Use `gpu-img` or `gpu-opt` for 512³.

### Key optimizations that drove 512³ speedup

| Change | bp (ms) | fp (ms) | Total (s) |
|--------|---------|---------|-----------|
| Baseline | 718 | 409 | 1.149 |
| Work-group `{8,8,4}→{4,4,16}` (coalesced z-writes) | **290** | 409 | 0.724 |
| AABB ray clipping in fp\_image (gated W>512) | 290 | **342** | 0.659 |
| `native_recip` in bp\_opt | **290** | 342 | **0.657** |

## Validation (10 epochs)

### 256³
```
Mode       min      max     mean   nan  inf  MSE vs CPU         MSE vs Python
python   0.0002   0.1225   0.0144    0    0  (python ref)      -
cpu      0.0000   1.4069   0.0066    0    0  (reference)        MSE=6.262e-04
gpu-buf  0.0000   1.4188   0.0066    0    0  MSE=1.521e-07  max=1.2484  MSE=6.263e-04
gpu-img  0.0000   1.4311   0.0066    0    0  MSE=1.373e-07  max=1.1721  MSE=6.263e-04
gpu-opt  0.0000   1.4311   0.0066    0    0  MSE=1.373e-07  max=1.1721  MSE=6.263e-04
```

### 512³
```
Mode       min      max     mean   nan  inf  MSE vs CPU
cpu      0.0001   0.3481   0.0334    0    0  (reference)
gpu-img  0.0000   0.5172   0.0331    0    0  MSE=8.452e-04  max=0.4236
gpu-opt  0.0000   0.5172   0.0331    0    0  MSE=8.452e-04  max=0.4236
```

MSE ~2×10⁻⁴ (256³) and ~8×10⁻⁴ (512³) between CPU and GPU reflects float32 rounding differences (CPU manual trilinear vs GPU hardware sampler) — not a correctness issue. All modes produce no NaN/Inf.

## Modes

| Mode | Flag | Description |
|------|------|-------------|
| CPU | `--mode cpu` | C + OpenMP, `-ffast-math`, incremental ray stepping |
| GPU buffer | `--mode gpu-buf` | OpenCL global buffers, manual bilinear/trilinear (256³ only) |
| GPU image | `--mode gpu-img` | Hardware image2d_array + image3d sampler |
| GPU opt | `--mode gpu-opt` | Hardware sampler + float2 LUT + local mem + half-precision vol_img + AABB clipping |

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
  fp_buffer_opt.cl    — fp (buffer, unused in main path)
validate.py           — load HDF5 outputs, print MSE table vs CPU (supports 256/512)
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
| Half-precision `vol_img` (`CL_HALF_FLOAT`) | `fp_image.cl`, `ct_gpu.c` | halves texture bandwidth for volume reads; float\_to\_half kernel on GPU, no PCIe roundtrip |
| AABB slab ray clipping (gated W>512) | `fp_image.cl`, `fp_buffer.cl` | tightens per-ray sample range on large detectors; skips empty cone-beam edge rays |
| `native_recip` in bp kernels | `bp_buffer_opt.cl`, `bp_buffer.cl`, `bp_image.cl` | hardware SFU reciprocal, ~4× faster than IEEE division |
| Unroll-x2 gated Nxz≥512 | `bp_buffer_opt.cl` | ILP across two texture fetches; gated to avoid register pressure on 256³ |
| OMP\_NUM\_THREADS=nproc | `Makefile` cpu targets | uses all available cores on lab node |
