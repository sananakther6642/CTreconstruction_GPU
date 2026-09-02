#!/usr/bin/env python3
"""Figure for the FP_TEX_EXACT precision result: default vs exact blending.

plot_results.py compares MODES against each other from the fixed
submission_outputs/ layout. This compares one mode against ITSELF with and
without FP_TEX_EXACT, from arbitrary directories, which that script has no
notion of.

Both difference panels are drawn on a SHARED colour scale, taken from the
default (worse) panel. This matters: plot_results.py's own diff maps
normalise each panel independently, which would rescale a 45x improvement
into looking identical. The whole point here is that one panel is visibly
emptier than the other.

The reconstructions themselves are deliberately not plotted side by side --
at 2.5e-09 vs 1.1e-07 the images are visually identical, so a slice
comparison would show two indistinguishable pictures and say nothing. The
error maps are where the result is actually visible.

Usage:
  python3 plot_fp_tex_exact.py --cpu REF.hdf5 --default D.hdf5 --exact E.hdf5 \
      [--mode gpu-img] [--scale 256] [--machine "NVIDIA GTX 680"] [--out NAME.png]
"""
import argparse
import sys

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py not found: pip install h5py")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    try:
        with h5py.File(path, "r") as f:
            return f["Volume"][:]
    except (OSError, KeyError) as e:
        sys.exit(f"cannot read {path}: {e}")


ap = argparse.ArgumentParser()
ap.add_argument("--cpu", required=True, help="CPU reference volume")
ap.add_argument("--default", required=True, dest="dflt",
                help="volume produced WITHOUT FP_TEX_EXACT")
ap.add_argument("--exact", required=True,
                help="volume produced WITH FP_TEX_EXACT=1")
ap.add_argument("--mode", default="gpu-img")
ap.add_argument("--scale", default="256")
ap.add_argument("--machine", default="NVIDIA GTX 680")
ap.add_argument("--out", default=None)
a = ap.parse_args()

ref = load(a.cpu).astype(np.float64)
d = load(a.dflt).astype(np.float64)
e = load(a.exact).astype(np.float64)
if not (ref.shape == d.shape == e.shape):
    sys.exit(f"shape mismatch: cpu{ref.shape} default{d.shape} exact{e.shape}")

mse_d = float(np.mean((d - ref) ** 2))
mse_e = float(np.mean((e - ref) ** 2))

mid = ref.shape[2] // 2
diff_d = d[:, :, mid] - ref[:, :, mid]
diff_e = e[:, :, mid] - ref[:, :, mid]

# Shared scale, set from a PERCENTILE of the default panel rather than its
# maximum. Using the max lets a handful of extreme voxels own the range and
# compresses the bulk of the error toward white in both panels -- which hides
# the very thing the figure exists to show, since the improvement is largely
# those extremes collapsing. The 99.5th percentile keeps typical error
# visible; the few voxels beyond it simply saturate.
vm = float(np.percentile(np.abs(diff_d), 99.5)) or 1e-12

fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
for ax, diff, mse, label in (
    (axes[0], diff_d, mse_d, "default (hardware blend)"),
    (axes[1], diff_e, mse_e, "FP_TEX_EXACT=1"),
):
    im = ax.imshow(diff, cmap="seismic", vmin=-vm, vmax=vm)
    ax.set_title(f"{label}\nMSE vs CPU = {mse:.3e}")
    ax.axis("off")

# ONE colourbar for both panels: two separate bars would suggest two
# independent scales and undercut the comparison the figure is making.
cb = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.02)
cb.set_label("difference vs CPU reference (saturating beyond ±99.5th pct)")

ratio = (mse_d / mse_e) if mse_e > 0 else float("inf")
fig.suptitle(
    f"{a.mode} error vs CPU reference, middle slice (z={mid}), "
    f"{a.scale}³, {a.machine}\n"
    f"identical colour scale on both panels; MSE reduced {ratio:.1f}×"
)
out = a.out or f"fp_tex_exact_{a.scale}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
print(f"  default MSE {mse_d:.4e}   exact MSE {mse_e:.4e}   ratio {ratio:.1f}x")
