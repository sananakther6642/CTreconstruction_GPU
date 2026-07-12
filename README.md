# CT Volume Reconstruction — GPU Lab Project

Cone-beam CT reconstruction using iterative MLEM update on CPU and GPU (OpenCL).

## Files

```
src/
  main.c            — argument parsing, timing, MSE reporting
  utils.c/h         — HDF5 load/save, timing
  ct_cpu.c/h        — CPU reference: bp, fp, cone-weight, iterative loop
  ct_gpu.c/h        — OpenCL host code: buffer, image, and optimized variants
kernels/
  bp_buffer.cl      — backprojection kernel (buffer)
  fp_buffer.cl      — forward projection kernel (buffer)
  bp_image.cl       — backprojection kernel (image2D array, hardware sampler)
  fp_image.cl       — forward projection kernel (image3D, hardware sampler)
  bp_buffer_opt.cl  — optimized bp: sin/cos LUT + local mem cache + atomics
  fp_buffer_opt.cl  — optimized fp: float3 stepping + 4x loop unroll + early exit
```

## Setup (lab machine — Linux with GPU)

### 1. Clone and switch to features branch
```bash
git clone https://github.com/sananakther6642/CTreconstruction_GPU.git
cd CTreconstruction_GPU
git checkout features
```

### 2. Install dependencies
```bash
sudo apt install libhdf5-dev ocl-icd-opencl-dev opencl-headers gcc make
```

Verify OpenCL sees your GPU:
```bash
clinfo | head -20
```

### 3. Build
```bash
make
```
Produces `build/ct_recon`.

### 4. Get hardware info (required for report)
```bash
grep "model name" /proc/cpuinfo | head -1
lspci | grep VGA
```

## Run

Data path `/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5` is the default — no need to pass `DATA=` on the lab machines.

```bash
# CPU reference
make run-cpu EPOCHS=100

# GPU buffer version
make run-gpu-buf EPOCHS=100

# GPU image version (hardware sampler)
make run-gpu-img EPOCHS=100

# GPU optimized version (LUT + local mem + float4 + loop unroll)
make run-gpu-opt EPOCHS=100
```

Quick test with 10 epochs first to check correctness:
```bash
make run-cpu EPOCHS=10
make run-gpu-buf EPOCHS=10
make run-gpu-img EPOCHS=10
make run-gpu-opt EPOCHS=10
```

Or run directly:
```bash
./build/ct_recon --data /lgrp/edu-2026-1-gpulab/proj_256_75.hdf5 \
                 --out output.hdf5 \
                 --mode gpu-opt \
                 --epochs 100 \
                 --kernels kernels/
```

Modes: `cpu`, `gpu-buf`, `gpu-img`, `gpu-opt`

## Validate (Python)

First run the Python reference to get ground truth:
```bash
# Edit out_path in Topic_2_CTreconstruction.py, then:
python3 Topic_2_CTreconstruction.py
```

Then compare all versions:
```bash
python3 - <<'EOF'
import h5py, numpy as np
from skimage.metrics import mean_squared_error

with h5py.File('output_python.hdf5')  as f: vp = f['Volume'][:]
with h5py.File('output_cpu.hdf5')     as f: vc = f['Volume'][:]
with h5py.File('output_gpu_buf.hdf5') as f: vb = f['Volume'][:]
with h5py.File('output_gpu_img.hdf5') as f: vi = f['Volume'][:]
with h5py.File('output_gpu_opt.hdf5') as f: vo = f['Volume'][:]

print(f"MSE cpu     vs python: {mean_squared_error(vp, vc):.6f}")
print(f"MSE gpu-buf vs python: {mean_squared_error(vp, vb):.6f}")
print(f"MSE gpu-img vs python: {mean_squared_error(vp, vi):.6f}")
print(f"MSE gpu-opt vs python: {mean_squared_error(vp, vo):.6f}")
EOF
```

## Algorithm

```
v0 = ones
for each epoch:
    b      = forward_project(v0)      # ray march through volume
    ratio  = p0 / b                   # measured / estimated
    v0    *= bp(ratio) / bp(ones)     # multiplicative update
```

Backprojection parallelism: one work-item per voxel `(ix, iy, iz)`.
Forward projection parallelism: one work-item per detector pixel `(iu, iv, ip)`.

## Optimizations (features branch)

| Optimization | Kernel | Effect |
|---|---|---|
| Sin/cos LUT (`float2 angle_cs`) | `bp_buffer_opt.cl`, `fp_buffer_opt.cl` | Eliminates trig per voxel/pixel |
| `__local` projection cache | `bp_buffer_opt.cl:bp_opt` | Cuts global mem reads by work-group size |
| Angle-parallel + atomic accumulate | `bp_buffer_opt.cl:bp_angle_parallel` | 4th dimension of parallelism |
| `float3` ray stepping + 4× unroll | `fp_buffer_opt.cl` | ILP, reduces loop overhead |
| Early OOB exit in trilinear | `fp_buffer_opt.cl` | Skips 8 checks for out-of-volume rays |
