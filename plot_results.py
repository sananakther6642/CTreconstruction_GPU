#!/usr/bin/env python3
"""
Report figures: OSEM convergence (log-likelihood vs wall-clock) and
reconstructed slice visualization. Run after run_convergence_sweep.sh
has produced conv_csv/osem_s*.csv, and after generating output_*.hdf5
via the normal make targets.

Usage:
  python3 plot_results.py convergence   # conv_csv/osem_s*.csv -> convergence.png
  python3 plot_results.py slices        # output_{cpu,gpu_buf,gpu_img,gpu_opt}.hdf5 -> slices.png
  python3 plot_results.py all
"""
import sys
import csv
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_convergence():
    configs = [1, 3, 5, 15, 25]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for s in configs:
        path = f"conv_csv/osem_s{s}.csv"
        times, logliks = [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                times.append(float(row["time_s"]))
                logliks.append(float(row["loglik"]))
        # time_s is per-epoch elapsed, not cumulative -- integrate
        cum_t = np.cumsum(times)
        label = "S=1 (plain MLEM)" if s == 1 else f"S={s}"
        ax.plot(cum_t, logliks, label=label, linewidth=1.6)

    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Poisson log-likelihood")
    ax.set_title("OSEM convergence: log-likelihood vs wall-clock (256³, 100 epochs)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("convergence.png", dpi=150)
    print("Saved: convergence.png")


def plot_slices():
    files = {
        "cpu":     "output_cpu.hdf5",
        "gpu-buf": "output_gpu_buf.hdf5",
        "gpu-img": "output_gpu_img.hdf5",
        "gpu-opt": "output_gpu_opt.hdf5",
    }
    vols = {}
    for name, path in files.items():
        with h5py.File(path, "r") as f:
            vols[name] = f["Volume"][:]

    ref = vols["cpu"]
    mid = ref.shape[2] // 2
    vmin, vmax = ref[:, :, mid].min(), ref[:, :, mid].max()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (name, vol) in zip(axes, vols.items()):
        im = ax.imshow(vol[:, :, mid], cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Reconstructed volume, middle slice (z={mid})")
    fig.tight_layout()
    fig.savefig("slices.png", dpi=150, bbox_inches="tight")
    print("Saved: slices.png")

    # difference maps vs CPU reference
    fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, name in zip(axes2, ["gpu-buf", "gpu-img", "gpu-opt"]):
        diff = ref[:, :, mid] - vols[name][:, :, mid]
        vm = max(abs(diff.min()), abs(diff.max())) or 1e-10
        im = ax.imshow(diff, cmap="seismic", vmin=-vm, vmax=vm)
        ax.set_title(f"cpu - {name}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig2.suptitle("Difference maps vs CPU reference (middle slice)")
    fig2.tight_layout()
    fig2.savefig("slices_diff.png", dpi=150, bbox_inches="tight")
    print("Saved: slices_diff.png")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("convergence", "all"):
        plot_convergence()
    if mode in ("slices", "all"):
        plot_slices()
