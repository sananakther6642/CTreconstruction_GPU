#!/usr/bin/env python3
"""Validate GPU reconstruction outputs against CPU reference."""
import sys
import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py not found: pip install h5py")

FILES = {
    'cpu':     'output_cpu.hdf5',
    'gpu-buf': 'output_gpu_buf.hdf5',
    'gpu-img': 'output_gpu_img.hdf5',
    'gpu-opt': 'output_gpu_opt.hdf5',
}

def load(path):
    try:
        with h5py.File(path, 'r') as f:
            return f['Volume'][:]
    except FileNotFoundError:
        return None

def stats(v):
    return (np.nanmin(v), np.nanmax(v), np.nanmean(v),
            int(np.sum(np.isnan(v))), int(np.sum(np.isinf(v))))

volumes = {k: load(p) for k, p in FILES.items()}

ref = volumes.get('cpu')
if ref is None:
    sys.exit("output_cpu.hdf5 not found — run cpu mode first")

print(f"{'Mode':<10} {'min':>10} {'max':>10} {'mean':>10} {'nan':>6} {'inf':>6}  MSE / max_diff")
print("-" * 75)

for name, v in volumes.items():
    if v is None:
        print(f"{name:<10}  [file not found]")
        continue
    mn, mx, me, nan_c, inf_c = stats(v)
    if name == 'cpu':
        print(f"{name:<10} {mn:>10.4f} {mx:>10.4f} {me:>10.4f} {nan_c:>6} {inf_c:>6}  (reference)")
    elif nan_c or inf_c:
        print(f"{name:<10} {mn:>10} {mx:>10} {me:>10} {nan_c:>6} {inf_c:>6}  DIVERGED")
    else:
        ref64 = ref.astype(np.float64)
        v64   = v.astype(np.float64)
        mse   = np.mean((v64 - ref64) ** 2)
        maxd  = np.max(np.abs(v64 - ref64))
        print(f"{name:<10} {mn:>10.4f} {mx:>10.4f} {me:>10.4f} {nan_c:>6} {inf_c:>6}  MSE={mse:.3e}  max_diff={maxd:.4f}")
