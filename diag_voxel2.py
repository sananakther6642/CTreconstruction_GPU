#!/usr/bin/env python3
"""Compare bp_cpu-equivalent math vs Python bp_func term-by-term at the
outlier voxel, using the ACTUAL measured projection data (not synthetic).
No singularity found in diag_voxel.py, so the divergence must come from
the projection data itself (cone_weight, interpolated pixel values) rather
than the backprojection geometry.
"""
import sys
import numpy as np
import h5py

sys.path.insert(0, '.')
from run_python_reference import bp_func, _get_pixelposition

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
    W = int(f['detector_width'][()])
    H = int(f['detector_height'][()])
    angles = f['Angle'][()]

cb_para = {
    'angles': angles, 'pixelSize': pixelSize, 'voxelSize': voxelSize,
    'Volumen_num_xz': Nxz, 'Volumen_num_y': Ny, 'SDD': SDD, 'SOD': SOD,
    'detector_width': W, 'detector_height': H,
}

out_ix, out_iy, iz = VOX
ix = Nxz - 1 - out_ix
iy = Nxz - 1 - out_iy
radius = Nxz * 0.5 - 0.5
radius_z = Ny * 0.5 - 0.5
xpr = (ix - radius) * voxelSize
ypr = (iy - radius) * voxelSize
zpr = (iz - radius_z) * voxelSize

print(f"voxel out=({out_ix},{out_iy},{iz})  pre-flip=({ix},{iy},{iz})")
print(f"xpr={xpr:.6f} ypr={ypr:.6f} zpr={zpr:.6f}")

# reproduce bp_func's exact preprocessing on ones (matches --op bp: cone_weight(ones))
ones_proj = np.ones((num_projs, H, W), dtype=np.float64)
proj = ones_proj.copy()[:, ::-1, :].transpose(0, 2, 1)  # [np][W][H]
proj /= voxelSize
a = -_get_pixelposition(W) * pixelSize
b = _get_pixelposition(H) * pixelSize
xu, yu = np.meshgrid(a, b, indexing='ij')
cone_weight = SDD / np.sqrt(SDD**2 + xu**2 + yu**2)
for i in range(num_projs):
    proj[i] *= cone_weight

from scipy.interpolate import RegularGridInterpolator

total = 0.0
per_angle = []
for ip, angle in enumerate(angles):
    ca, sa = np.cos(angle), np.sin(angle)
    t = ypr * ca - xpr * sa
    U = SOD + ypr * sa + xpr * ca
    ai = SDD * t / U
    bi = zpr * SDD / U
    interp = RegularGridInterpolator((a, b), proj[ip], bounds_error=False, fill_value=0)
    val = float(interp((ai, bi))) * (SOD**2) / (U**2)
    total += val
    per_angle.append((ip, np.degrees(angle), U, ai, bi, val))

total *= np.pi / num_projs
print(f"\nPython-style bp(ones) at this voxel: {total:.6f}")

# top contributors
per_angle.sort(key=lambda r: -abs(r[5]))
print(f"\n{'ip':>4} {'deg':>8} {'U':>10} {'ai':>10} {'bi':>10} {'contrib':>12}")
for ip, deg, U, ai, bi, val in per_angle[:10]:
    print(f"{ip:4d} {deg:8.2f} {U:10.4f} {ai:10.4f} {bi:10.4f} {val:12.4f}")

print(f"\nsum of top-10 |contrib| as %total: "
      f"{100*sum(abs(v) for *_, v in per_angle[:10])/max(abs(total),1e-9):.1f}%")

# compare to what CPU op=bp dump actually has at this voxel, if available
try:
    with h5py.File('bp_cpu.hdf5', 'r') as f:
        cpu_vol = f['Volume'][:]
    print(f"\nC bp_cpu.hdf5 value at output voxel {VOX}: {cpu_vol[VOX]:.6f}")
except FileNotFoundError:
    print("\n(bp_cpu.hdf5 not found — run: make run-op-bp)")
