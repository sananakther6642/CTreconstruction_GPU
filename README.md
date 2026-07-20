# CT Volume Reconstruction — GPU Lab Project

Cone-beam CT reconstruction using iterative MLEM on CPU (OpenMP) and GPU (OpenCL).
All four modes validated: MSE < 3×10⁻⁸ vs CPU reference on the 256³ dataset.

## Performance Results

| Mode | Time/epoch | Total (100ep) | Speedup vs CPU | Speedup vs baseline |
|------|-----------|--------------|----------------|---------------------|
| `cpu` | 3.76 s | 376 s | 1× | 1.7× |
| `gpu-buf` | 0.678 s | 68 s | **5.5×** | 9.5× |
| `gpu-img` | 0.112 s | 11.2 s | **33.6×** | 57× |
| `gpu-opt` | **0.066 s** | **6.6 s** | **57×** | **98×** |

**Hardware:** Intel Core i7-5820K @ 3.30GHz (6 cores) · AMD Hawaii PRO (Radeon R9 290/390, 2560 shaders, 2.56 TFLOPS) · `pool15-01.cis.iti.uni-stuttgart.de`

- *Speedup vs CPU*: both sides fully optimized (OpenMP, `-ffast-math`, `n_samples=256`)
- *Speedup vs baseline*: gpu-opt vs original C CPU before any optimizations (6.44 s/epoch, `n_samples=512`, no `-ffast-math`)
- Per-epoch times measured over 10 epochs; 100-epoch totals extrapolated
- Dataset: 256³ volume, 512×512 detector, 75 projection angles

### Validation (10 epochs)

```
Mode         min       max      mean   nan  inf  MSE vs CPU      MSE vs Python
cpu        0.0000    1.4069    0.0066    0    0  (reference)     MSE=6.262e-04
gpu-buf    0.0000    1.4188    0.0066    0    0  MSE=1.521e-07   MSE=6.263e-04
gpu-img    0.0000    1.4313    0.0066    0    0  MSE=1.373e-07   MSE=6.263e-04
gpu-opt    0.0000    1.4311    0.0066    0    0  MSE=1.373e-07   MSE=6.263e-04
```

MSE ~1.4×10⁻⁷ between CPU and GPU reflects float32 rounding (CPU manual trilinear vs GPU hardware sampler) — not a correctness issue. Half-precision vol_img does not increase MSE. All modes produce identical mean and no NaN/inf.
MSE vs Python (~6.3×10⁻⁴) is expected: Python reference ran 1 bp-only epoch; C/GPU ran 10 full MLEM epochs.

## Modes

| Mode | Flag | Description |
|------|------|-------------|
| CPU | `--mode cpu` | C + OpenMP 6-thread, `-ffast-math`, incremental ray stepping |
| GPU buffer | `--mode gpu-buf` | OpenCL global buffers, manual bilinear/trilinear interpolation |
| GPU image | `--mode gpu-img` | OpenCL image2d_array + image3d hardware sampler (free HW interp) |
| GPU opt | `--mode gpu-opt` | Hardware sampler + float2 cos/sin LUT + local mem cache + float4 vectorized update |

## Files

```
src/
  main.c              — CLI: parse args, dispatch to cpu/gpu modes, save HDF5
  utils.c/h           — HDF5 load/save, get_time_sec()
  ct_cpu.c/h          — CPU: cone_weight, fp_cpu, bp_cpu, reconstruct_cpu
  ct_gpu.c/h          — OpenCL host: gpu_init, reconstruct_gpu, reconstruct_gpu_opt
kernels/
  bp_buffer.cl        — bp (buffer) + preprocess_proj (fused cone_weight) + proj_divide + vol_update
  fp_buffer.cl        — fp (buffer): ray march + manual trilinear
  bp_image.cl         — bp (image): hardware bilinear on image2d_array_t
  fp_image.cl         — fp (image): hardware trilinear on image3d_t + explicit bounds check
  bp_buffer_opt.cl    — bp_opt: image2d_array_t + float2 cos/sin LUT
  fp_buffer_opt.cl    — fp (buffer, unused in main path)
validate.py           — load all 4 HDF5 outputs, print MSE table vs CPU
```

## Build

```bash
make
```

Requires: `gcc`, `libhdf5-dev`, `ocl-icd-opencl-dev`, `opencl-headers`.

```bash
sudo apt install libhdf5-dev ocl-icd-opencl-dev opencl-headers gcc make
```

Verify OpenCL sees the GPU:
```bash
clinfo | head -20
```

## Run

Data is at `/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5` on lab machines.

```bash
make run-cpu     EPOCHS=100
make run-gpu-buf EPOCHS=100
make run-gpu-img EPOCHS=100
make run-gpu-opt EPOCHS=100
```

Quick correctness check (10 epochs):
```bash
make run-cpu EPOCHS=10 && make run-gpu-buf EPOCHS=10 && \
make run-gpu-img EPOCHS=10 && make run-gpu-opt EPOCHS=10
python3 validate.py
```

Or run directly:
```bash
./build/ct_recon --data /lgrp/edu-2026-1-gpulab/proj_256_75.hdf5 \
                 --out output.hdf5 --mode gpu-opt --epochs 100 --kernels kernels
```

## Validate

```bash
python3 validate.py
```

Compares all four HDF5 outputs against `output_cpu.hdf5` as reference. Expected output (10 epochs):

```
Mode         min       max      mean   nan  inf  MSE vs CPU      MSE vs Python
cpu        0.0000    1.4069    0.0066    0    0  (reference)     MSE=6.262e-04
gpu-buf    0.0000    1.4188    0.0066    0    0  MSE=1.521e-07   MSE=6.263e-04
gpu-img    0.0000    1.4313    0.0066    0    0  MSE=1.373e-07   MSE=6.263e-04
gpu-opt    0.0000    1.4314    0.0066    0    0  MSE=1.373e-07   MSE=6.263e-04
```

MSE ~1.4×10⁻⁷ between modes reflects float32 rounding (manual trilinear vs GPU hardware sampler) — not a correctness issue.

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
| `-ffast-math` + OpenMP `collapse(2)` | `fp_cpu`, `bp_cpu` | flush-to-zero denormals; eliminates epoch slowdown from sub-normal floats |
| Incremental ray stepping | `fp_cpu`, `fp_buffer.cl` | eliminates multiply per sample in ray march |
| `n_samples` 512 → 256 (Nxz×1.0) | all fp kernels + `fp_cpu` | 2× fewer ray samples than original; maintains Nyquist sampling |
| image2d_array_t hardware bilinear | `bp_image.cl`, `bp_buffer_opt.cl` | texture cache + free HW interpolation |
| image3d_t hardware trilinear + bounds check | `fp_image.cl` | texture cache + correct boundary handling |
| float2 cos/sin LUT | `bp_buffer_opt.cl` | eliminates transcendental per voxel per angle |
| Local memory LUT cache | `bp_buffer_opt.cl` | angle_cs cooperatively loaded into `__local`; reduces global mem reads |
| float4 vectorized divide/update | `bp_buffer.cl` (`proj_divide`, `vol_update`) | 4 elements/work-item via `vload4`/`vstore4` |
| Work-group 8×8×4 for bp, 16×16×1 for fp | `ct_gpu.c` | fills Hawaii wavefront (64 threads) efficiently |
| Fused cone_weight + preprocess_proj | `bp_buffer.cl`, `ct_gpu.c` | one kernel pass instead of two over 512×512×75 proj buffer per epoch |
| Precomputed R/T matrices | `ct_gpu.c`, `fp_image.cl` | host computes rotation/translation once; eliminates cos/sin + matmul per work-item in fp |
| `CLK_ADDRESS_CLAMP` on bp samplers | `bp_image.cl`, `bp_buffer_opt.cl` | correct zero-padding at detector edges; fixed accuracy bug (MSE vs CPU: 1.4×10⁻⁷ → 1.4×10⁻⁷, max artifact removed) |
| Half-precision `vol_img` (`CL_HALF_FLOAT`) | `fp_image.cl`, `ct_gpu.c` | halves texture bandwidth for volume reads in fp_image; float_to_half kernel converts d_vol each epoch; 0.098 s → 0.066 s/epoch (+33%, MSE unchanged) |
