# CT Volume Reconstruction — GPU Lab Project

Cone-beam CT reconstruction using iterative MLEM on CPU (OpenMP) and GPU (OpenCL).
All four modes validated: MSE < 3×10⁻⁸ vs CPU reference on the 256³ dataset.

## Performance Results

| Mode | Time/epoch | Total (100ep) | Speedup vs CPU | Speedup vs baseline |
|------|-----------|--------------|----------------|---------------------|
| `cpu` | 2.45 s | 245 s | 1× | 2.6× |
| `gpu-buf` | 0.54 s | 54 s | **4.5×** | 11.9× |
| `gpu-img` | 0.083 s | 8.3 s | **29.5×** | 77.6× |
| `gpu-opt` | **0.070 s** | **7.0 s** | **35×** | **92×** |

**Hardware:** Intel Core i7-5820K @ 3.30GHz (6 cores) · AMD Hawaii PRO (Radeon R9 290/390, 2560 shaders, 2.56 TFLOPS) · `pool15-01.cis.iti.uni-stuttgart.de`

- *Speedup vs CPU*: both sides fully optimized (OpenMP, `-ffast-math`, `n_samples=128`)
- *Speedup vs baseline*: gpu-opt vs original C CPU before any optimizations (6.44 s/epoch, `n_samples=512`, no `-ffast-math`)
- Per-epoch times measured over 10 epochs; 100-epoch totals extrapolated
- Dataset: 256³ volume, 512×512 detector, 75 projection angles

### Validation (10 epochs)

```
Mode         min       max      mean   nan  inf  MSE vs CPU      MSE vs Python
cpu        0.0000    3.2045    0.0067    0    0  (reference)     MSE=6.654e-04
gpu-buf    0.0000    3.2050    0.0067    0    0  MSE=2.882e-08   MSE=6.653e-04
gpu-img    0.0000    3.2083    0.0067    0    0  MSE=2.975e-08   MSE=6.652e-04
gpu-opt    0.0000    3.2082    0.0067    0    0  MSE=2.988e-08   MSE=6.652e-04
```

MSE ~3×10⁻⁸ between CPU and GPU reflects float32 rounding (CPU manual trilinear vs GPU hardware sampler) — not a correctness issue. All modes produce identical mean and no NaN/inf.
MSE vs Python (~6.65×10⁻⁴) is expected: Python reference ran 1 bp-only epoch; C/GPU ran 10 full MLEM epochs.

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
cpu        0.0000    3.2045    0.0067    0    0  (reference)     MSE=6.654e-04
gpu-buf    0.0000    3.2050    0.0067    0    0  MSE=2.882e-08   MSE=6.653e-04
gpu-img    0.0000    3.2083    0.0067    0    0  MSE=2.975e-08   MSE=6.652e-04
gpu-opt    0.0000    3.2082    0.0067    0    0  MSE=2.988e-08   MSE=6.652e-04
```

MSE ~3×10⁻⁸ between modes reflects float32 rounding (manual trilinear vs GPU hardware sampler) — not a correctness issue.

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
| `n_samples` 512 → 256 → 128 (Nxz×0.5) | all fp kernels + `fp_cpu` | 4× fewer ray samples than original; fp time quartered |
| Fused cone_weight into preprocess_proj | `bp_buffer.cl`, `ct_cpu.c` | eliminates separate kernel dispatch per epoch across all modes |
| image2d_array_t hardware bilinear | `bp_image.cl`, `bp_buffer_opt.cl` | texture cache + free HW interpolation |
| image3d_t hardware trilinear + bounds check | `fp_image.cl` | texture cache + correct boundary handling |
| float2 cos/sin LUT | `bp_buffer_opt.cl` | eliminates transcendental per voxel per angle |
| Local memory LUT cache | `bp_buffer_opt.cl` | angle_cs cooperatively loaded into `__local`; reduces global mem reads |
| float4 vectorized divide/update | `bp_buffer.cl` (`proj_divide`, `vol_update`) | 4 elements/work-item via `vload4`/`vstore4` |
| Work-group 8×8×4 for bp, 16×16×1 for fp | `ct_gpu.c` | fills Hawaii wavefront (64 threads) efficiently |
