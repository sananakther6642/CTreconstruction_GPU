#!/usr/bin/env python3
"""Validate GPU reconstruction outputs against CPU reference.
Usage:
  python3 validate.py        # 256^3 outputs (default)
  python3 validate.py 512    # 512^3 outputs
"""
import sys
import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py not found: pip install h5py")

suffix = '_512' if (len(sys.argv) > 1 and sys.argv[1] == '512') else ''

FILES = {
    'python':  'output_python.hdf5',
    'cpu':     f'output_cpu{suffix}.hdf5',
    'gpu-buf': f'output_gpu_buf{suffix}.hdf5',
    'gpu-img': f'output_gpu_img{suffix}.hdf5',
    'gpu-opt': f'output_gpu_opt{suffix}.hdf5',
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
    sys.exit(f"{FILES['cpu']} not found — run cpu mode first")

py_ref = volumes.get('python')

label = "512^3" if suffix else "256^3"
print(f"\n=== {label} validation ===")
print(f"\n{'Mode':<10} {'min':>10} {'max':>10} {'mean':>10} {'nan':>6} {'inf':>6}  MSE vs CPU     MSE vs Python")
print("-" * 95)

for name, v in volumes.items():
    if v is None:
        if name == 'python':
            continue
        print(f"{name:<10}  [file not found]")
        continue

    mn, mx, me, nan_c, inf_c = stats(v)

    if name == 'cpu':
        py_mse = ""
        if py_ref is not None:
            m = np.mean((v.astype(np.float64) - py_ref.astype(np.float64))**2)
            py_mse = f"MSE={m:.3e}"
        print(f"{name:<10} {mn:>10.4f} {mx:>10.4f} {me:>10.4f} {nan_c:>6} {inf_c:>6}  (reference)    {py_mse}")
    elif name == 'python':
        print(f"{name:<10} {mn:>10.4f} {mx:>10.4f} {me:>10.4f} {nan_c:>6} {inf_c:>6}  (python ref)   -")
    elif nan_c or inf_c:
        print(f"{name:<10} {mn:>10} {mx:>10} {me:>10} {nan_c:>6} {inf_c:>6}  DIVERGED")
    else:
        ref64 = ref.astype(np.float64)
        v64   = v.astype(np.float64)
        mse   = np.mean((v64 - ref64)**2)
        maxd  = np.max(np.abs(v64 - ref64))
        cpu_str = f"MSE={mse:.3e}  max={maxd:.4f}"
        py_str = ""
        if py_ref is not None:
            m = np.mean((v64 - py_ref.astype(np.float64))**2)
            py_str = f"MSE={m:.3e}"
        print(f"{name:<10} {mn:>10.4f} {mx:>10.4f} {me:>10.4f} {nan_c:>6} {inf_c:>6}  {cpu_str:<20} {py_str}")

print()
