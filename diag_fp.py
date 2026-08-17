#!/usr/bin/env python3
"""Compare fp_cpu(ones_volume) vs Python fp_func(ones_volume), full projection
stack and at specific pixels. bp(ones) was already verified clean (agrees to
5 decimals) — this checks whether fp is where the max=1.17 outlier actually
comes from.

Usage:
  make run-op-fp   # generates fp_cpu.hdf5
  python3 diag_fp.py
"""
import sys
import numpy as np
import h5py

sys.path.insert(0, '.')
from run_python_reference import fp_func

DATA = sys.argv[1] if len(sys.argv) > 1 else '/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5'

with h5py.File(DATA, 'r') as f:
    voxelSize = float(f['voxelSize'][()])
    Nxz = int(f['Volumen_num_xz'][()])
    Ny = int(f['Volumen_num_y'][()])
    SDD = float(f['SDD'][()])
    SOD = float(f['SOD'][()])
    pixelSize = float(f['pixelSize'][()])
    num_projs = int(f['num_projs'][()])
    W = int(f['detector_width'][()])
    H = int(f['detector_height'][()])
    angles = f['Angle'][()]

cb_para = {
    'angles': angles, 'pixelSize': pixelSize, 'voxelSize': voxelSize,
    'Volumen_num_xz': Nxz, 'Volumen_num_y': Ny, 'SDD': SDD, 'SOD': SOD,
    'detector_width': W, 'detector_height': H,
}

with h5py.File('fp_cpu.hdf5', 'r') as f:
    c_proj = f['Projection'][:].astype(np.float64)  # [np, H, W]

print(f"Loaded C fp_cpu.hdf5: shape={c_proj.shape}")

# CPU's --op fp uses n_samples=Nxz by default (no --samples flag passed to
# `make run-op-fp`); fp_func's n_samples = ceil(Nxz * sample_ratio), so
# sample_ratio must be 1.0 here to match, not the module's default of 2.
sample_ratio = 1.0
print(f"Running Python fp_func(ones, sample_ratio={sample_ratio}, jitter=False) "
      f"-> n_samples={int(np.ceil(Nxz*sample_ratio))} (matches C's default n_samples=Nxz={Nxz})...")

ones_vol = np.ones((Nxz, Nxz, Nxz), dtype=np.float32)
py_proj = fp_func(cb_para, ones_vol, sample_ratio=sample_ratio, jitter=False)  # [np, H, W]
py_proj = py_proj.astype(np.float64)

print(f"Python fp_func: shape={py_proj.shape}")

if py_proj.shape != c_proj.shape:
    sys.exit(f"SHAPE MISMATCH: python={py_proj.shape} vs C={c_proj.shape} "
             f"— check --samples matches sample_ratio*Nxz (n_samples default = Nxz={Nxz})")

diff = np.abs(c_proj - py_proj)
mse = np.mean(diff**2)
maxd = np.max(diff)
idx = np.unravel_index(np.argmax(diff), diff.shape)
print(f"\nfp MSE={mse:.4e}  max_abs_diff={maxd:.4e}  worst_pixel(ip,ih,iw)={idx}")
print(f"  C value at worst pixel:      {c_proj[idx]:.6f}")
print(f"  Python value at worst pixel: {py_proj[idx]:.6f}")

# is the outlier concentrated at detector edges (would point to a boundary/
# AABB clipping mismatch) or spread through the middle (would point to
# sampling/ray-march math)?
ip, ih, iw = idx
print(f"\n  detector edge distance: ih={ih}/{H} (dist_to_edge={min(ih,H-1-ih)}), "
      f"iw={iw}/{W} (dist_to_edge={min(iw,W-1-iw)})")

n_outliers = int(np.sum(diff > 10*np.sqrt(mse))) if mse > 0 else 0
print(f"  outliers (diff>10*rms): {n_outliers} / {diff.size} pixels "
      f"({100*n_outliers/diff.size:.4f}%)")
