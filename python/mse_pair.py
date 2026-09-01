#!/usr/bin/env python3
"""MSE between any two reconstruction volumes.

validate.py always scores against the CPU volume, which is the right
reference for the headline "MSE vs CPU" number but forces a CPU run
(~4.2 s/epoch at 256^3, ~10x gpu-buf) even when CPU is not the thing being
measured.

For isolating the hardware sampler specifically, gpu-buf is the better and
much cheaper reference: it is the manual float32 interpolation path, so
MSE(gpu-img, gpu-buf) is exactly the sampler's own contribution, with the
CPU/GPU toolchain difference divided out on both sides.

Usage:
    python3 mse_pair.py <a.hdf5> <b.hdf5> [label]
"""
import sys

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py not found: pip install h5py")

if len(sys.argv) < 3:
    sys.exit(__doc__)

a_path, b_path = sys.argv[1], sys.argv[2]
label = sys.argv[3] if len(sys.argv) > 3 else ""


def load(path):
    try:
        with h5py.File(path, "r") as f:
            return f["Volume"][:]
    except (OSError, KeyError) as e:
        sys.exit(f"cannot read {path}: {e}")


a, b = load(a_path), load(b_path)
if a.shape != b.shape:
    sys.exit(f"shape mismatch: {a.shape} vs {b.shape}")

a64 = a.astype(np.float64)
b64 = b.astype(np.float64)
diff = np.abs(a64 - b64)
mse = float(np.mean(diff ** 2))
maxd = float(diff.max())

# Signal range of the reference, so RMS can be read as a % the way the
# README's existing accuracy bullets report it.
rng = float(b64.max() - b64.min())
rms_pct = (mse ** 0.5 / rng * 100.0) if rng > 0 else float("nan")

print(
    f"{label:<24} MSE={mse:.4e}  max|diff|={maxd:.4e}  "
    f"RMS={rms_pct:.4f}% of range  "
    f"({a_path.split('/')[-1]} vs {b_path.split('/')[-1]})"
)
