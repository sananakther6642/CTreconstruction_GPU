#!/usr/bin/env python3
"""
Report figures: OSEM convergence, plain-MLEM convergence (all modes,
both dataset sizes), and reconstructed slice visualization (both
sizes). Reads from submission_outputs/{kale,pool15}/, the archived
results layout (each source self-contained: conv_csv/ + hdf5 outputs).

Usage:
  python3 plot_results.py osem                        # kale, 256^3 OSEM sweep only (pool15 has no OSEM data)
  python3 plot_results.py mlem   [--source pool15]
  python3 plot_results.py slices [--source pool15] [--scale 256|512]
  python3 plot_results.py all    [--source pool15]

--source kale (default): submission_outputs/kale/conv_csv/mlem_{scale}_{mode}.csv,
                          submission_outputs/kale/output_{mode}{_512}.hdf5
--source pool15:         submission_outputs/pool15/conv_csv/{mode}_{scale}.csv,
                          submission_outputs/pool15/{mode}_{scale}.hdf5
Output filenames get a suffix matching --source (kale runs stay
unsuffixed for backward compatibility with earlier report drafts).
"""
import argparse
import csv
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODES = ["cpu", "gpu-buf", "gpu-img", "gpu-opt"]


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


def _suffix(source):
    return "" if source == "kale" else f"_{source}"


def _mlem_csv_path(source, scale, mode):
    if source == "kale":
        return f"submission_outputs/kale/conv_csv/mlem_{scale}_{mode}.csv"
    return f"submission_outputs/{source}/conv_csv/{mode}_{scale}.csv"


def _hdf5_path(source, scale, mode):
    if source == "kale":
        suffix = "" if scale == "256" else "_512"
        return f"submission_outputs/kale/output_{mode.replace('-', '_')}{suffix}.hdf5"
    return f"submission_outputs/{source}/{mode}_{scale}.hdf5"


def plot_osem_convergence(source="kale"):
    if source != "kale":
        print("OSEM sweep is kale/gpu-opt/256^3 only (pool15-01 has no OSEM "
              "data) -- skipping osem for --source", source)
        return
    configs = [1, 3, 5, 15, 25]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for s in configs:
        cum_t, logliks = _read_csv(f"submission_outputs/kale/conv_csv/osem_s{s}.csv")
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


def plot_mlem_convergence(source="kale"):
    suf = _suffix(source)
    hw_label = "AMD Hawaii PRO / pool15-01" if source == "pool15" else "kale"
    for scale in ("256", "512"):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for mode in MODES:
            path = _mlem_csv_path(source, scale, mode)
            try:
                cum_t, logliks = _read_csv(path)
            except FileNotFoundError:
                print(f"  missing {path}, skipping {mode} in this plot")
                continue
            ax.plot(cum_t, logliks, label=mode, linewidth=1.6)
        ax.set_xlabel("Wall-clock time (s)")
        ax.set_ylabel("Poisson log-likelihood")
        ax.set_title(f"Plain MLEM convergence, all modes ({scale}³, 100 epochs, {hw_label})")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = f"mlem_convergence_{scale}{suf}.png"
        fig.savefig(out, dpi=150)
        print(f"Saved: {out}")


def plot_slices(source="kale", scale="256"):
    suf = _suffix(source)
    hw_label = "AMD Hawaii PRO / pool15-01" if source == "pool15" else "kale"
    vols = {}
    for mode in MODES:
        path = _hdf5_path(source, scale, mode)
        try:
            with h5py.File(path, "r") as f:
                vols[mode] = f["Volume"][:]
        except FileNotFoundError:
            print(f"  missing {path}, skipping slices for {scale}^3 ({source})")
            return

    ref = vols["cpu"]
    mid = ref.shape[2] // 2
    vmin, vmax = ref[:, :, mid].min(), ref[:, :, mid].max()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (name, vol) in zip(axes, vols.items()):
        im = ax.imshow(vol[:, :, mid], cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Reconstructed volume, middle slice (z={mid}), {scale}³, {hw_label}")
    fig.tight_layout()
    out = f"slices_{scale}{suf}.png"
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
    fig2.suptitle(f"Difference maps vs CPU reference (middle slice), {scale}³, {hw_label}")
    fig2.tight_layout()
    out2 = f"slices_diff_{scale}{suf}.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["osem", "mlem", "slices", "all"], default="all", nargs="?")
    parser.add_argument("--source", default="kale", choices=["kale", "pool15"],
                         help="which machine's data to plot (default: kale)")
    parser.add_argument("--scale", default=None, choices=["256", "512"],
                         help="for 'slices': only this scale (default: both)")
    args = parser.parse_args()

    if args.what in ("osem", "all"):
        plot_osem_convergence(args.source)
    if args.what in ("mlem", "all"):
        plot_mlem_convergence(args.source)
    if args.what in ("slices", "all"):
        scales = [args.scale] if args.scale else ["256", "512"]
        for scale in scales:
            plot_slices(args.source, scale)
