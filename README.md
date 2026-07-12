# CT Volume Reconstruction — GPU Lab Project

Cone-beam CT reconstruction using iterative MLEM update on CPU and GPU (OpenCL).

## Files

```
src/
  main.c       — argument parsing, timing, MSE reporting
  utils.c/h    — HDF5 load/save, timing
  ct_cpu.c/h   — CPU reference: bp, fp, cone-weight, iterative loop
  ct_gpu.c/h   — OpenCL host code: buffer and image variants
kernels/
  bp_buffer.cl — backprojection kernel (buffer)
  fp_buffer.cl — forward projection kernel (buffer)
  bp_image.cl  — backprojection kernel (image2D array)
  fp_image.cl  — forward projection kernel (image3D)
```

## Build

```bash
make
```

Requires: `gcc`, `libhdf5-dev`, OpenCL headers + ICD loader.

On lab machines (Linux):
```bash
sudo apt install libhdf5-dev ocl-icd-opencl-dev opencl-headers
```

Check GPU/CPU info for the report:
```bash
grep name /proc/cpuinfo | head -1
lspci | grep VGA
```

## Run

```bash
# CPU reference
make run-cpu   DATA=/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5 EPOCHS=100

# GPU buffer version
make run-gpu-buf

# GPU image version
make run-gpu-img
```

Or directly:
```bash
./build/ct_recon --data /lgrp/edu-2026-1-gpulab/proj_256_75.hdf5 \
                 --out output.hdf5 \
                 --mode gpu-buf \
                 --epochs 100 \
                 --kernels kernels/
```

Modes: `cpu`, `gpu-buf`, `gpu-img`

## Validate (Python)

```python
import h5py, numpy as np
from skimage.metrics import mean_squared_error

with h5py.File('output_python.hdf5') as f: v_py  = f['Volume'][:]
with h5py.File('output_cpu.hdf5')    as f: v_cpu = f['Volume'][:]
with h5py.File('output_gpu_buf.hdf5')as f: v_gpu = f['Volume'][:]

print('MSE cpu  vs python:', mean_squared_error(v_py, v_cpu))
print('MSE gpu  vs python:', mean_squared_error(v_py, v_gpu))
```

## Algorithm

```
v0 = ones
for each epoch:
    b      = forward_project(v0)      # ray march through volume
    ratio  = p0 / b                   # measured / estimated
    v0    *= bp(ratio) / bp(ones)     # multiplicative update
```

Backprojection parallelism: one work-item per voxel (ix, iy, iz).  
Forward projection parallelism: one work-item per detector pixel (iu, iv, ip).
