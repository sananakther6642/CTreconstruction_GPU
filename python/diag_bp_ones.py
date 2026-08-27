#!/usr/bin/env python3
"""
Diagnostic for the 26-voxel outlier spike found in
output_python_reconstruction.hdf5 (Topic_2_CTreconstruction.py, 256^3,
20 epochs) -- MSE-vs-Topic2 was 2.06e-02 with these voxels in, 3.2e-04
excluding them, so they dominate the reported number. All 26 sit at
z=1 or z=254 (one slice in from the physical edge, not the edge itself)
or a small interior cluster.

Hypothesis: Topic_2_CTreconstruction.py's MLEM update divides by
bp_ones with no zero-guard (line ~253: `ratio = bp_f(result) / bp_ones`,
unlike the nearby np.divide(..., where=(b != 0)) which does guard).
If bp_ones is near-zero at those voxels, this diverges under the
multiplicative update over 20 epochs.

This script computes bp_ones only (one bp_func call, not the full
MLEM loop) and reports its value at the 26 known outlier voxel
indices plus overall min/histogram, to confirm or rule out the
near-zero theory directly rather than guessing from the final output.

Run on kale (needs skimage + the real dataset path).
"""
import sys
import numpy as np
import h5py

sys.path.insert(0, ".")
from Topic_2_CTreconstruction import bp_func

OUTLIER_VOXELS = [
    (0, 25, 1), (0, 26, 254), (0, 57, 1), (0, 218, 1), (0, 252, 202),
    (39, 255, 1), (68, 255, 254), (72, 0, 254), (77, 0, 1), (84, 255, 1),
    (90, 52, 6), (98, 119, 2), (147, 255, 254), (173, 26, 11),
    (243, 60, 9), (243, 60, 10), (243, 60, 11), (243, 61, 6), (243, 61, 7),
    (243, 255, 1), (246, 67, 12), (249, 185, 19),
    (255, 5, 254), (255, 6, 254), (255, 71, 1), (255, 192, 1),
]

path_data = "/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5"

with h5py.File(path_data, "r") as f:
    voxelSize = f["voxelSize"][()]
    Volumen_num_xz = int(f["Volumen_num_xz"][()])
    Volumen_num_y = int(f["Volumen_num_y"][()])
    SDD = f["SDD"][()]
    SOD = f["SOD"][()]
    pixelSize = f["pixelSize"][()]
    num_projs = int(f["num_projs"][()])
    detector_width = int(f["detector_width"][()])
    detector_height = int(f["detector_height"][()])
    angles = f["Angle"][()]
    projection_0 = f["Projection"][:, :, :]

cb_para = {
    "angles": angles,
    "pixelSize": pixelSize,
    "voxelSize": voxelSize,
    "Volumen_num_xz": Volumen_num_xz,
    "Volumen_num_y": Volumen_num_y,
    "SDD": SDD,
    "SOD": SOD,
    "detector_width": detector_width,
    "detector_height": detector_height,
}

print("Computing bp_ones (single bp_func call on ones)...")
bp_ones = bp_func(np.ones_like(projection_0), cb_para)

print(f"\nbp_ones shape: {bp_ones.shape}")
print(f"bp_ones global min/max/mean: {bp_ones.min()} / {bp_ones.max()} / {bp_ones.mean()}")
print(f"bp_ones count == 0: {(bp_ones == 0).sum()}")
print(f"bp_ones count < 1e-6: {(bp_ones < 1e-6).sum()}")
print(f"bp_ones count < 1e-3: {(bp_ones < 1e-3).sum()}")

print("\n=== bp_ones value at each known outlier voxel ===")
for idx in OUTLIER_VOXELS:
    val = bp_ones[idx]
    print(f"  {idx}: bp_ones={val:.6e}")

print("\n=== bp_ones at a random sample of NON-outlier voxels, for comparison ===")
rng = np.random.default_rng(0)
for _ in range(10):
    idx = tuple(rng.integers(0, [256, 256, 256]))
    if idx in OUTLIER_VOXELS:
        continue
    print(f"  {idx}: bp_ones={bp_ones[idx]:.6e}")

print("\n=== z-slice min(bp_ones) profile near z=0..5 and z=250..255 ===")
for z in list(range(0, 6)) + list(range(250, 256)):
    print(f"  z={z}: min={bp_ones[:,:,z].min():.6e}  mean={bp_ones[:,:,z].mean():.6e}")
