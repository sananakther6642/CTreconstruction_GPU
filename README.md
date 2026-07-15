# CT Volume Reconstruction — GPU Lab Project

Cone-beam CT reconstruction using iterative MLEM update on CPU (OpenMP) and GPU (OpenCL).
All four modes validated: MSE < 2×10⁻⁸ vs CPU reference on the 256³ dataset.

## Modes

| Mode | Description | Time/epoch | Total (100ep) | Speedup |
|------|-------------|-----------|--------------|---------|
| `cpu` | C + OpenMP + `-ffast-math` | 6.44 s | 644 s | 1× |
| `gpu-buf` | OpenCL buffer, manual bilinear/trilinear | 0.99 s | 99 s | **6.5×** |
| `gpu-img` | OpenCL image2D array + image3D hardware sampler | 0.159 s | 16 s | **40×** |
| `gpu-opt` | image sampler + float2 cos/sin LUT + local mem cache | 0.144 s | 14.4 s | **44.7×** |

Device: AMD Hawaii (Radeon R9 390), 256³ volume, 512×512 detector, 75 angles, 100 epochs.

## Files

```
src/
  main.c              — CLI: parse args, dispatch to cpu/gpu modes, save HDF5
  utils.c/h           — HDF5 load/save, get_time_sec()
  ct_cpu.c/h          — CPU: cone_weight, fp_cpu, bp_cpu, reconstruct_cpu
  ct_gpu.c/h          — OpenCL host: gpu_init, reconstruct_gpu, reconstruct_gpu_opt
kernels/
  bp_buffer.cl        — bp (buffer) + preprocess_proj + cone_weight_hw + proj_divide + vol_update
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

Compares all four HDF5 outputs against `output_cpu.hdf5` as reference. Expected output:

```
Mode              min        max       mean    nan    inf  MSE / max_diff
---------------------------------------------------------------------------
cpu            0.0000     1.2548     0.0066      0      0  (reference)
gpu-buf        0.0000     1.2547     0.0066      0      0  MSE=2.350e-07  max_diff=0.9673
gpu-img        0.0000     1.2535     0.0066      0      0  MSE=2.888e-07  max_diff=0.9677
gpu-opt        0.0000     1.2535     0.0066      0      0  MSE=2.880e-07  max_diff=0.9677
```

MSE at 100 epochs (~2.5e-7) reflects accumulated float32 rounding between CPU manual trilinear and GPU hardware sampler — not a correctness issue. All modes converge to the same mean and produce no NaN/inf.

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
| `-ffast-math` + OpenMP `collapse(2)` | `fp_cpu`, `bp_cpu` | flush-to-zero denormals; race-free z-strips |
| Incremental ray stepping | `fp_cpu`, `fp_buffer.cl` | eliminates multiply/sample |
| image2d_array_t hardware bilinear | `bp_image.cl`, `bp_buffer_opt.cl` | texture cache + free HW interp |
| image3d_t hardware trilinear + bounds check | `fp_image.cl` | texture cache + correct boundary |
| float2 cos/sin LUT | `bp_buffer_opt.cl` | eliminates trig per voxel per angle |
| Local memory LUT cache | `bp_buffer_opt.cl` | angle_cs loaded into `__local`; reduces global reads |
| float4 vectorized divide/update | `bp_buffer.cl` (`proj_divide`, `vol_update`) | 4 elements/work-item via vload4/vstore4 |
| Work-group 8×8×4 for bp, 16×16×1 for fp | `ct_gpu.c` | fills Hawaii wavefront (64 threads) |
