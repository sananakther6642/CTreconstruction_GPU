#!/usr/bin/env python3
"""Diagnose why voxel (49,211,32) is the persistent CPU/GPU-vs-Python outlier.

Checks whether U = SOD + ypr*sin(angle) + xpr*cos(angle) approaches 0 for any
angle at this voxel (a geometric near-singularity that would blow up 1/U^2
and amplify tiny float32 rounding differences between implementations),
and prints the actual bp_cpu-style math step by step so we can compare
against what the CPU/GPU code computes.

RESULT (measured, 256^3, see diag_maxgap.py): the singularity hypothesis this
script was written to test is FALSE. min|U| over all 75 angles at the worst
voxel is 4.10 -- 82% of SOD -- and max 1/U^2 is only 1.5x its isocentre
value. No amplification worth speaking of. The real cpu-vs-gpu-img/gpu-opt
disagreement is a one-voxel displacement of sharp features from hardware
CLK_FILTER_LINEAR sampling, not geometry. Kept because the U-vs-angle dump
is still the right tool if a genuine singularity is ever suspected.
"""
import sys
import numpy as np
import h5py

DATA = sys.argv[1] if len(sys.argv) > 1 else '/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5'
VOX = tuple(int(x) for x in sys.argv[2].split(',')) if len(sys.argv) > 2 else (49, 211, 32)

with h5py.File(DATA, 'r') as f:
    voxelSize = float(f['voxelSize'][()])
    Nxz = int(f['Volumen_num_xz'][()])
    Ny = int(f['Volumen_num_y'][()])
    SDD = float(f['SDD'][()])
    SOD = float(f['SOD'][()])
    pixelSize = float(f['pixelSize'][()])
    num_projs = int(f['num_projs'][()])
    detector_width = int(f['detector_width'][()])
    detector_height = int(f['detector_height'][()])
    angles = f['Angle'][()]

print(f"voxelSize={voxelSize}  Nxz={Nxz}  Ny={Ny}  SDD={SDD}  SOD={SOD}  pixelSize={pixelSize}")
print(f"detector: {detector_width} x {detector_height}  num_projs={num_projs}")

# NOTE: validate.py's worst_voxel_idx is the numpy array index into the
# OUTPUT volume, i.e. AFTER the [::-1,::-1,:] flip both bp_cpu (Nxz-1-ix,
# Nxz-1-iy) and bp_func ([::-1,::-1,:]) apply. To get the pre-flip (ix,iy,iz)
# that the inner loop actually iterated over, invert the flip.
out_ix, out_iy, iz = VOX
ix = Nxz - 1 - out_ix
iy = Nxz - 1 - out_iy
print(f"\noutput-space voxel: {VOX}")
print(f"pre-flip loop voxel (ix,iy,iz): ({ix},{iy},{iz})")

radius = Nxz * 0.5 - 0.5
radius_z = Ny * 0.5 - 0.5
xpr = (ix - radius) * voxelSize
ypr = (iy - radius) * voxelSize
zpr = (iz - radius_z) * voxelSize
print(f"xpr={xpr:.4f}  ypr={ypr:.4f}  zpr={zpr:.4f}  (mm)")

print(f"\n{'angle':>8} {'U':>12} {'1/U^2':>14} {'ai(mm)':>10} {'bi(mm)':>10} {'in-bounds?':>10}")
a_min = (detector_width - 1) * 0.5 * pixelSize  # matches C: a_min = (W-1)/2*px
b_min = -(detector_height - 1) * 0.5 * pixelSize
worst_u = None
for ip, angle in enumerate(angles):
    ca, sa = np.cos(angle), np.sin(angle)
    U = SOD + ypr * sa + xpr * ca
    t = ypr * ca - xpr * sa
    ai = SDD * t / U
    bi = zpr * SDD / U
    # detector bounds check (mm -> pixel), matches CPU's uf/vf logic roughly
    uf_px = (ai - (-(0 - (detector_width - 1) * 0.5) * pixelSize)) / (-pixelSize)
    in_bounds = 0 <= uf_px < detector_width - 1
    if worst_u is None or abs(U) < abs(worst_u):
        worst_u = U
    if abs(U) < SOD * 0.15 or ip < 3 or ip > num_projs - 3:
        print(f"{np.degrees(angle):8.2f} {U:12.5f} {1/(U*U):14.2f} {ai:10.3f} {bi:10.3f} {str(in_bounds):>10}")

print(f"\nmin |U| across all angles: {worst_u:.5f}  (SOD={SOD})")
print(f"max 1/U^2 across all angles: {max(1/((SOD+ypr*np.sin(a)+xpr*np.cos(a))**2) for a in angles):.2f}")
