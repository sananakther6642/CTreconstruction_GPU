#!/usr/bin/env python3
"""Per-operator validation: compare a single fp() or bp() call against the
Python reference, isolated from accumulated MLEM iteration.

Why this exists: validate.py compares final volumes after N epochs of MLEM,
so a wrong operator and 10 epochs of compounding error look the same in the
final MSE. This isolates which kernel (fp vs bp) is actually wrong.

Usage:
  # 1. generate the CPU component-test dump:
  ./build/ct_recon --data <data.hdf5> --out fp_cpu.hdf5 --mode cpu --op fp --kernels kernels
  ./build/ct_recon --data <data.hdf5> --out bp_cpu.hdf5 --mode cpu --op bp --kernels kernels

  # 2. compare against Python reference:
  python3 validate_ops.py bp --data /lgrp/edu-2026-1-gpulab/proj_256_75.hdf5 --dump bp_cpu.hdf5
  python3 validate_ops.py fp --data /lgrp/edu-2026-1-gpulab/proj_256_75.hdf5 --dump fp_cpu.hdf5
"""
import sys
import argparse
import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py not found: pip install h5py")

sys.path.insert(0, '.')
from Topic_2_CTreconstruction import bp_func, fp_func  # noqa: E402


def load_cb_para(data_path):
    with h5py.File(data_path, 'r') as f:
        cb_para = {
            'angles': f['Angle'][()],
            'pixelSize': f['pixelSize'][()],
            'voxelSize': f['voxelSize'][()],
            'Volumen_num_xz': int(f['Volumen_num_xz'][()]),
            'Volumen_num_y': int(f['Volumen_num_y'][()]),
            'SDD': f['SDD'][()],
            'SOD': f['SOD'][()],
            'detector_width': int(f['detector_width'][()]),
            'detector_height': int(f['detector_height'][()]),
        }
        num_projs = int(f['num_projs'][()])
    return cb_para, num_projs


def compare(name, ref, out):
    ref64, out64 = ref.astype(np.float64), out.astype(np.float64)
    mse = np.mean((ref64 - out64) ** 2)
    maxd = np.max(np.abs(ref64 - out64))
    print(f"{name:<20} shape={out.shape}  MSE={mse:.4e}  max_abs_diff={maxd:.4e}  "
          f"ref_mean={ref64.mean():.4e}  out_mean={out64.mean():.4e}")
    return mse, maxd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('op', choices=['fp', 'bp'])
    ap.add_argument('--data', required=True, help='path to proj_*.hdf5 dataset')
    ap.add_argument('--dump', required=True, help='HDF5 dump from ./ct_recon --op fp|bp')
    args = ap.parse_args()

    cb_para, num_projs = load_cb_para(args.data)

    with h5py.File(args.dump, 'r') as f:
        key = 'Projection' if args.op == 'fp' else 'Volume'
        c_out = f[key][:]

    if args.op == 'bp':
        print("Running Python bp_func(cone_weight(ones)) ...")
        ones_proj = np.ones((num_projs, cb_para['detector_height'],
                              cb_para['detector_width']), dtype=np.float32)
        py_ref = bp_func(ones_proj, cb_para)
    else:
        print("Running Python fp_func(ones_volume) ... (slow, may take minutes)")
        nvxz = cb_para['Volumen_num_xz']
        ones_vol = np.ones((nvxz, nvxz, nvxz), dtype=np.float32)
        py_ref = fp_func(cb_para, ones_vol, sample_ratio=2)

    if py_ref.shape != c_out.shape:
        sys.exit(f"shape mismatch: python={py_ref.shape} vs C={c_out.shape} "
                  f"— check --samples matches sample_ratio*Nxz")

    print()
    compare(f"C {args.op}_cpu vs python", py_ref, c_out)


if __name__ == '__main__':
    main()
