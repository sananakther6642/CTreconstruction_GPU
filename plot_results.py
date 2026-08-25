#!/usr/bin/env python3
"""
Report figures: OSEM convergence, plain-MLEM convergence (all modes,
both dataset sizes), and reconstructed slice visualization (both
sizes). Run after run_convergence_sweep.sh has produced conv_csv/*.csv,
and after generating output_*.hdf5 / output_*_512.hdf5 via the normal
make targets.

Usage:
  python3 plot_results.py osem        # conv_csv/osem_s*.csv -> convergence.png / convergence_full.png
  python3 plot_results.py mlem        # conv_csv/mlem_*.csv -> mlem_convergence_256.png / _512.png
  python3 plot_results.py slices      # output_*.hdf5 -> slices_256.png / slices_diff_256.png
  python3 plot_results.py slices512   # output_*_512.hdf5 -> slices_512.png / slices_diff_512.png
  python3 plot_results.py all
"""
import sys
import csv
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _read_csv(path):
    times, logliks = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            times.append(float(row["time_s"]))
            logliks.append(float(row["loglik"]))
    cum_t = np.cumsum(times)
    # epoch 1 is a cold-start value that dwarfs the converged range on a
    # linear axis -- drop it, matching the OSEM report table's convention.
    return cum_t[1:], logliks[1:]


def plot_osem_convergence():
    configs = [1, 3, 5, 15, 25]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for s in configs:
        cum_t, logliks = _read_csv(f"conv_csv/osem_s{s}.csv")
        label = "S=1 (plain MLEM)" if s == 1 else f"S={s}"
        ax.plot(cum_t, logliks, label=label, linewidth=1.6)

    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Poisson log-likelihood")
    ax.set_title("OSEM convergence: log-likelihood vs wall-clock (256³, 100 epochs)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("convergence_full.png", dpi=150)
    print("Saved: convergence_full.png")

    ax.set_xlim(0, 40)
    fig.savefig("convergence.png", dpi=150)
    print("Saved: convergence.png")


def plot_mlem_convergence():
    modes = ["cpu", "gpu-buf", "gpu-img", "gpu-opt"]
    for scale in ("256", "512"):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for mode in modes:
            key = mode.replace("-", "_")
            path = f"conv_csv/mlem_{scale}_{mode}.csv"
            cum_t, logliks = _read_csv(path)
            ax.plot(cum_t, logliks, label=mode, linewidth=1.6)
        ax.set_xlabel("Wall-clock time (s)")
        ax.set_ylabel("Poisson log-likelihood")
        ax.set_title(f"Plain MLEM convergence, all modes ({scale}³, 100 epochs)")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = f"mlem_convergence_{scale}.png"
        fig.savefig(out, dpi=150)
        print(f"Saved: {out}")


def plot_slices(scale="256"):
    suffix = "" if scale == "256" else "_512"
    files = {
        "cpu":     f"output_cpu{suffix}.hdf5",
        "gpu-buf": f"output_gpu_buf{suffix}.hdf5",
        "gpu-img": f"output_gpu_img{suffix}.hdf5",
        "gpu-opt": f"output_gpu_opt{suffix}.hdf5",
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
    fig.suptitle(f"Reconstructed volume, middle slice (z={mid}), {scale}³")
    fig.tight_layout()
    out = f"slices_{scale}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, name in zip(axes2, ["gpu-buf", "gpu-img", "gpu-opt"]):
        diff = ref[:, :, mid] - vols[name][:, :, mid]
        vm = max(abs(diff.min()), abs(diff.max())) or 1e-10
        im = ax.imshow(diff, cmap="seismic", vmin=-vm, vmax=vm)
        ax.set_title(f"cpu - {name}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig2.suptitle(f"Difference maps vs CPU reference (middle slice), {scale}³")
    fig2.tight_layout()
    out2 = f"slices_diff_{scale}.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("osem", "all"):
        plot_osem_convergence()
    if mode in ("mlem", "all"):
        plot_mlem_convergence()
    if mode in ("slices", "all"):
        plot_slices("256")
    if mode in ("slices512", "all"):
        plot_slices("512")
