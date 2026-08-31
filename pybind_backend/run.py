"""
Reference call site for the pybind11 CT reconstruction backend.

Run from the repo root:

    python3 -m pybind_backend.run --mode gpu-opt --epochs 100

or directly:

    python3 pybind_backend/run.py --mode cpu --epochs 20
"""
import argparse
import ctypes
import os

import h5py
import numpy as np

# Force large allocations (the 64 MiB volume, the 75 MiB projection array)
# to come from a fresh mmap rather than glibc's fragmented main arena.
# The CLI (src/main.c:119) mallocs the volume in a near-empty process, so
# it always lands on a clean anonymous mapping; here, torch + h5py + the
# projection read have already churned the heap by the time the volume is
# allocated. M_MMAP_THRESHOLD is glibc's -3; setting it below 64 MiB makes
# numpy's allocator behave like the CLI's for these buffers. Best-effort:
# non-glibc or a missing symbol just leaves the default in place.
try:
    ctypes.CDLL("libc.so.6").mallopt(-3, 32 * 1024 * 1024)
except (OSError, AttributeError):
    pass

from pybind_backend.backend import KERNEL_DIR, _backend

# Matches the Makefile's DATA256 default -- the shared dataset path on
# the lab machines, not a path relative to the repo (a fresh clone has
# no proj_256_75.hdf5 sitting in its working directory).
DEFAULT_DATA = "/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5"


def _scalar(f, key):
    """load_hdf5/save_hdf5 (src/utils.c) store scalars as shape-(1,)
    datasets, not true 0-d scalars. float(f[key][()]) on such an array
    raises a numpy DeprecationWarning today and will hard-error on a
    future numpy ("Conversion of an array with ndim > 0 to a scalar is
    deprecated"). Reshape-and-index instead."""
    return np.asarray(f[key][()]).reshape(-1)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=DEFAULT_DATA, help="input HDF5 path")
    ap.add_argument("--out", default="output_py.hdf5", help="output HDF5 path")
    ap.add_argument(
        "--mode",
        choices=["cpu", "gpu-buf", "gpu-img", "gpu-opt"],
        default="gpu-opt",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--subsets", type=int, default=1, help="gpu-opt only")
    ap.add_argument("--samples", type=int, default=0, help="0 = auto (Nxz)")
    ap.add_argument("--half", action="store_true")
    args = ap.parse_args()

    with h5py.File(args.data, "r") as f:
        # read_direct instead of [:].astype(float32): the latter
        # materializes the full native-dtype array first and then a second
        # float32 copy, leaving a large freed hole in the heap that the
        # volume allocation below can land in.
        proj = np.empty(f["Projection"].shape, dtype=np.float32)
        f["Projection"].read_direct(proj)
        angles = f["Angle"][:].astype(np.float64)
        voxelSize = float(_scalar(f, "voxelSize"))
        pixelSize = float(_scalar(f, "pixelSize"))
        SDD = float(_scalar(f, "SDD"))
        SOD = float(_scalar(f, "SOD"))
        Nxz = int(_scalar(f, "Volumen_num_xz"))
        Ny = int(_scalar(f, "Volumen_num_y"))
        W = int(_scalar(f, "detector_width"))
        H = int(_scalar(f, "detector_height"))
        num_projs = int(_scalar(f, "num_projs"))

    common = dict(
        proj=proj,
        angles=angles,
        voxelSize=voxelSize,
        pixelSize=pixelSize,
        SDD=SDD,
        SOD=SOD,
        Nxz=Nxz,
        Ny=Ny,
        W=W,
        H=H,
        num_projs=num_projs,
        epochs=args.epochs,
        n_samples=args.samples,
        use_half=args.half,
    )

    if args.mode == "cpu":
        volume = _backend.reconstruct_cpu(**common)
    elif args.mode == "gpu-buf":
        volume = _backend.reconstruct_gpu_buf(kernel_dir=KERNEL_DIR, **common)
    elif args.mode == "gpu-img":
        volume = _backend.reconstruct_gpu_img(kernel_dir=KERNEL_DIR, **common)
    else:  # gpu-opt
        volume = _backend.reconstruct_gpu_opt(
            kernel_dir=KERNEL_DIR, subsets=args.subsets, **common
        )

    with h5py.File(args.out, "w") as f:
        f.create_dataset("voxelSize", data=voxelSize)
        f.create_dataset("Volume", data=volume)

    print(f"Wrote {args.out}  (mode={args.mode}, shape={volume.shape})")


if __name__ == "__main__":
    main()
