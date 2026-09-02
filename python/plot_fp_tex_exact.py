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

# ── Why this figure is built the way it is ─────────────────────────────
# A plain two-panel difference map badly undersells this result. At 256^3
# the error is fine-grained speckle, and MSE squares it, so the 44.7x is
# driven by a small number of extreme voxels that occupy almost no pixels.
# The eye reads typical speckle amplitude, which changes far less. The
# reference figure submission_outputs/hawaii/slices_diff_256_hawaii.png
# shows the opposite failure: per-panel autoscaling makes a ~980x MSE gap
# between gpu-buf and gpu-img look like no gap at all, because each panel
# is normalised to its own range.
#
# So: two shared-scale maps for spatial structure, plus a log-scale
# histogram of |error| that shows the actual distribution shift. The
# histogram is where the magnitude becomes readable -- it puts the tail,
# which is what MSE responds to, on an axis where a 45x change is visible.
vm = float(np.percentile(np.abs(diff_d), 99.9)) or 1e-12

fig = plt.figure(figsize=(15.5, 5.4))
# Extra wspace: the shared colourbar sits to the right of the second map,
# and at tighter spacing it overlaps the histogram's y-axis label.
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.55)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
for ax, diff, mse, label in (
    (ax0, diff_d, mse_d, "default (hardware blend)"),
    (ax1, diff_e, mse_e, "FP_TEX_EXACT=1"),
):
    im = ax.imshow(diff, cmap="seismic", vmin=-vm, vmax=vm)
    ax.set_title(f"{label}\nMSE vs CPU = {mse:.3e}", fontsize=10)
    ax.axis("off")

# One colourbar for both maps: separate bars would imply separate scales,
# which is precisely the flaw in the reference figure above.
cb = fig.colorbar(im, ax=[ax0, ax1], fraction=0.030, pad=0.04)
cb.set_label("difference vs CPU reference", fontsize=9)

# Error-magnitude distribution, whole volume (not just the shown slice).
ax2 = fig.add_subplot(gs[0, 2])
ad = np.abs(d - ref).ravel()
ae = np.abs(e - ref).ravel()
lo = max(min(ad[ad > 0].min(), ae[ae > 0].min()), 1e-12)
hi = max(ad.max(), ae.max())
bins = np.logspace(np.log10(lo), np.log10(hi), 120)
ax2.hist(ad, bins=bins, histtype="step", label="default", color="crimson")
ax2.hist(ae, bins=bins, histtype="step", label="FP_TEX_EXACT=1", color="steelblue")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("|error| vs CPU", fontsize=9)
ax2.set_ylabel("voxel count", fontsize=9)
ax2.set_title("error distribution, whole volume\n(log-log; the tail drives MSE)",
              fontsize=10)
ax2.legend(fontsize=8, loc="upper right")
ax2.tick_params(labelsize=8)

ratio = (mse_d / mse_e) if mse_e > 0 else float("inf")
fig.suptitle(
    f"{a.mode} error vs CPU reference, {a.scale}³, {a.machine}   —   "
    f"maps share one colour scale, middle slice (z={mid});   MSE reduced {ratio:.1f}×",
    y=0.98, fontsize=11,
)
fig.subplots_adjust(top=0.80)
out = a.out or f"fp_tex_exact_{a.scale}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
print(f"  default MSE {mse_d:.4e}   exact MSE {mse_e:.4e}   ratio {ratio:.1f}x")
