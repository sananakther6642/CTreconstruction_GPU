#!/usr/bin/env python3
"""What does bp_ones actually look like, and is the FOV-edge amplification
theory even possible?

The FOV-mask sweep found rel=1e-4, 1e-3 (and possibly 1e-2) bit-identical
to rel=0 -- i.e. NO voxel has bp_ones below that fraction of the peak. That
contradicts the theory that near-zero bp_ones amplifies CPU/GPU differences
at FOV-edge voxels. This dumps the real distribution so the theory can be
confirmed or discarded on evidence instead of another sweep arm.

Prints, for the sensitivity map:
  - percentiles of bp_ones/max, so the true dynamic range is visible
  - bp_ones at the specific worst-diff voxel the validator keeps naming
  - how much amplification 1/bp_ones actually delivers there

Run where the .hdf5 outputs live, after a --dump-bp-ones run (see below).
"""
import sys
import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py not found: pip install h5py")

if len(sys.argv) < 2:
    sys.exit("usage: diag_bp_ones.py <bp_ones.hdf5> [ix iy iz]")

with h5py.File(sys.argv[1], "r") as f:
    key = "Volume" if "Volume" in f else list(f.keys())[0]
    bo = f[key][:].astype(np.float64)

mx = bo.max()
print(f"shape={bo.shape}  max={mx:.6g}  min={bo.min():.6g}  mean={bo.mean():.6g}")
print(f"zeros: {int((bo == 0).sum())}   below 1e-10: {int((bo < 1e-10).sum())}")

print("\nbp_ones / max(bp_ones) percentiles:")
for q in [0, 0.001, 0.01, 0.1, 1, 5, 25, 50, 100]:
    v = np.percentile(bo, q) / mx
    print(f"  p{q:<7} {v:.6e}")

print("\nfraction of voxels below a given rel threshold "
      "(what FOV_MASK_REL would actually mask):")
for rel in [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1]:
    frac = float((bo < rel * mx).mean())
    print(f"  rel={rel:<7g} masks {frac*100:.4f}% of voxels")

# The validator repeatedly names (49,212,33) as gpu-img's worst voxel.
ix, iy, iz = (int(a) for a in (sys.argv[2:5] or (49, 212, 33)))
v = bo[ix, iy, iz]
print(f"\nworst-diff voxel ({ix},{iy},{iz}): bp_ones={v:.6g}  "
      f"= {v/mx:.6e} of max")
print(f"  amplification 1/bp_ones = {1.0/v:.6g}" if v > 0 else "  bp_ones is ZERO")
