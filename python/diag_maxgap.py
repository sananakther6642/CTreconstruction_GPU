#!/usr/bin/env python3
"""Explain the 256^3 max-value gap between cpu (1.7303) and gpu-img/gpu-opt
(1.8741) reported by validate.py.

validate.py reports a per-mode `max` over the whole volume and a `max` abs
diff vs CPU. Those can come from different voxels, so this checks, in order:

  1. Where is each mode's global max? Same voxel or different ones?
  2. At the worst-diff voxel, what do all four modes actually hold, and what
     do the immediate neighbours look like (isolated spike vs smooth region)?
  3. Is that voxel at the FOV-edge geometric singularity (small |U| for some
     angle => 1/U^2 blows up and amplifies float32 differences)?
  4. How many voxels are actually affected, and where do they sit radially?

Run on kale/pool15 where the .hdf5 outputs live:
    python3 diag_maxgap.py                     # 256^3, default files
    python3 diag_maxgap.py 512                 # 512^3
"""
import sys
import numpy as np
import h5py

scale = sys.argv[1] if len(sys.argv) > 1 else '256'
suffix = '_512' if scale == '512' else ''
DATA = (f'/lgrp/edu-2026-1-gpulab/proj_{scale}_75.hdf5')

FILES = {
    'cpu':     f'output_cpu{suffix}.hdf5',
    'gpu-buf': f'output_gpu_buf{suffix}.hdf5',
    'gpu-img': f'output_gpu_img{suffix}.hdf5',
    'gpu-opt': f'output_gpu_opt{suffix}.hdf5',
}

vols = {}
for name, path in FILES.items():
    try:
        with h5py.File(path, 'r') as f:
            vols[name] = f['Volume'][:]
    except (OSError, KeyError) as e:
        sys.exit(f"cannot read {path}: {e}")

ref = vols['cpu']
Nxz, _, Ny = ref.shape
print(f"volume shape: {ref.shape}")

# ---- 1. where is each mode's global max? -------------------------------
print("\n=== 1. global max location per mode ===")
print(f"{'mode':<9} {'max':>10} {'at voxel':>20}")
for name, v in vols.items():
    idx = np.unravel_index(np.argmax(v), v.shape)
    print(f"{name:<9} {v.max():>10.4f} {str(idx):>20}")

# ---- 2. worst-diff voxel and its neighbourhood -------------------------
print("\n=== 2. worst |gpu-opt - cpu| voxel and neighbourhood ===")
diff = np.abs(vols['gpu-opt'].astype(np.float64) - ref.astype(np.float64))
w = np.unravel_index(np.argmax(diff), diff.shape)
print(f"worst voxel: {w}   |diff| = {diff[w]:.4f}")
print(f"{'mode':<9} {'value here':>12}")
for name, v in vols.items():
    print(f"{name:<9} {v[w]:>12.4f}")

print("\nimmediate neighbours along each axis (cpu / gpu-opt):")
for ax, label in enumerate('xyz'):
    lo = list(w); hi = list(w)
    lo[ax] = max(0, w[ax] - 1)
    hi[ax] = min(ref.shape[ax] - 1, w[ax] + 1)
    lo, hi = tuple(lo), tuple(hi)
    print(f"  {label}-1 {lo}: {ref[lo]:8.4f} / {vols['gpu-opt'][lo]:8.4f}"
          f"    {label}+1 {hi}: {ref[hi]:8.4f} / {vols['gpu-opt'][hi]:8.4f}")

# ---- 3. geometry at that voxel ----------------------------------------
print("\n=== 3. geometry check (is |U| small for some angle?) ===")
try:
    with h5py.File(DATA, 'r') as f:
        voxelSize = float(f['voxelSize'][()])
        SOD = float(f['SOD'][()])
        SDD = float(f['SDD'][()])
        angles = f['Angle'][()]
except OSError as e:
    print(f"  (dataset not readable here: {e} -- skipping geometry)")
else:
    # validate.py indexes the OUTPUT volume, which bp writes flipped
    # ([::-1,::-1,:]); invert to get the loop-space voxel the math used.
    ix = Nxz - 1 - w[0]
    iy = Nxz - 1 - w[1]
    iz = w[2]
    radius = Nxz * 0.5 - 0.5
    radius_z = Ny * 0.5 - 0.5
    xpr = (ix - radius) * voxelSize
    ypr = (iy - radius) * voxelSize
    zpr = (iz - radius_z) * voxelSize
    U = SOD + ypr * np.sin(angles) + xpr * np.cos(angles)
    inv = 1.0 / (U * U)
    print(f"  pre-flip loop voxel (ix,iy,iz) = ({ix},{iy},{iz})")
    print(f"  xpr={xpr:.4f} ypr={ypr:.4f} zpr={zpr:.4f} mm   SOD={SOD}")
    print(f"  min |U| over 75 angles = {np.abs(U).min():.5f}"
          f"   ({100*np.abs(U).min()/SOD:.1f}% of SOD)")
    print(f"  max 1/U^2              = {inv.max():.2f}"
          f"   (vs {1.0/(SOD*SOD):.2f} at the isocentre)")
    print(f"  amplification vs isocentre = {inv.max()*SOD*SOD:.1f}x")

# ---- 4. how widespread is it? -----------------------------------------
print("\n=== 4. spread of the disagreement ===")
mse = float(np.mean(diff ** 2))
rms = np.sqrt(mse)
rng = float(ref.max() - ref.min())
print(f"  MSE = {mse:.3e}   RMS = {rms:.3e}   signal range = {rng:.4f}")
print(f"  RMS as % of range = {100*rms/rng:.4f}%")
for k in (10, 100, 1000):
    n = int(np.sum(diff > k * rms))
    print(f"  voxels with |diff| > {k:>4}*RMS: {n:>10}  ({100.0*n/diff.size:.6f}% of volume)")

# radial position of the worst offenders
thr = 100 * rms
bad = np.argwhere(diff > thr)
if len(bad):
    c = (Nxz - 1) / 2.0
    r = np.sqrt((bad[:, 0] - c) ** 2 + (bad[:, 1] - c) ** 2)
    print(f"\n  {len(bad)} voxels above 100*RMS:")
    print(f"    radial distance from axis: min={r.min():.1f}  "
          f"median={np.median(r):.1f}  max={r.max():.1f}   (FOV radius = {Nxz/2:.0f})")
    print(f"    fraction beyond 0.8*FOV radius: "
          f"{100.0*np.sum(r > 0.8*Nxz/2)/len(r):.1f}%")
