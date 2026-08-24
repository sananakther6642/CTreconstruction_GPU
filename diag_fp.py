#!/usr/bin/env python3
"""Compare fp_cpu(ones_volume) vs Python fp_func(ones_volume), full projection
stack and at specific pixels. bp(ones) was already verified clean (agrees to
5 decimals) — this checks whether fp is where the max=1.17 outlier actually
comes from.

Usage:
  make run-op-fp       # generates fp_cpu.hdf5 (256^3)
  python3 diag_fp.py

  make run-op-fp-512   # generates fp_cpu_512.hdf5 (512^3, SAMPLES512 samples)
  python3 diag_fp.py --data /lgrp/edu-2026-1-gpulab/proj_512_75.hdf5 \\
                      --dump fp_cpu_512.hdf5 --samples 512
"""
import sys
import argparse
import numpy as np
import h5py

sys.path.insert(0, '.')
from run_python_reference import fp_func

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5')
ap.add_argument('--dump', default='fp_cpu.hdf5',
                 help='HDF5 dump from ./ct_recon --op fp')
ap.add_argument('--samples', type=int, default=None,
                 help='n_samples used for the C dump (must match --samples '
                      'passed to ct_recon, or omit if it used the default '
                      'n_samples=Nxz)')
args = ap.parse_args()
DATA = args.data

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

with h5py.File(args.dump, 'r') as f:
    c_proj = f['Projection'][:].astype(np.float64)  # [np, H, W]

print(f"Loaded C dump {args.dump}: shape={c_proj.shape}")

# CPU's --op fp uses n_samples=Nxz by default when --samples is omitted;
# fp_func's n_samples = ceil(Nxz * sample_ratio), so sample_ratio must be
# chosen to reproduce whatever n_samples the C dump actually used — pass
# --samples explicitly if the C run used --samples (e.g. 512^3 uses
# SAMPLES512, which happens to equal Nxz=512 by default but don't assume
# that holds for other overrides).
n_samples_used = args.samples if args.samples is not None else Nxz
sample_ratio = n_samples_used / Nxz
print(f"Running Python fp_func(ones, sample_ratio={sample_ratio:.6f}, jitter=False) "
      f"-> n_samples={int(np.ceil(Nxz*sample_ratio))} (matches C's n_samples={n_samples_used})...")

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
