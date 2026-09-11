#!/usr/bin/env python3
"""
Report figures: OSEM convergence, plain-MLEM convergence (all modes,
both dataset sizes), and reconstructed slice visualization (both
sizes). Reads from
results/{intel_xeon_e5_2620_nvidia_geforce_gtx_680,intel_i7_5820k_amd_hawaii_pro}/,
the archived results layout (each source self-contained: conv_csv/ +
hdf5 outputs).

Usage:
  python3 plot_results.py osem                                                    # Platform B, 256^3 OSEM sweep only (Platform A has no OSEM data)
  python3 plot_results.py mlem   [--source intel_i7_5820k_amd_hawaii_pro]
  python3 plot_results.py slices [--source intel_i7_5820k_amd_hawaii_pro] [--scale 256|512]
  python3 plot_results.py all    [--source intel_i7_5820k_amd_hawaii_pro]

--source intel_xeon_e5_2620_nvidia_geforce_gtx_680 (default, Platform B):
    results/intel_xeon_e5_2620_nvidia_geforce_gtx_680/conv_csv/mlem_{scale}_{mode}.csv,
    results/intel_xeon_e5_2620_nvidia_geforce_gtx_680/output_{mode}{_512}.hdf5
--source intel_i7_5820k_amd_hawaii_pro (Platform A):
    results/intel_i7_5820k_amd_hawaii_pro/conv_csv/{mode}_{scale}.csv,
    results/intel_i7_5820k_amd_hawaii_pro/{mode}_{scale}.hdf5
Output filenames get a suffix matching --source (Platform B runs stay
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


PLATFORM_B = "intel_xeon_e5_2620_nvidia_geforce_gtx_680"
PLATFORM_A = "intel_i7_5820k_amd_hawaii_pro"


def _suffix(source):
    return "" if source == PLATFORM_B else f"_{source}"


def _mlem_csv_path(source, scale, mode):
    if source == PLATFORM_B:
        return f"results/{PLATFORM_B}/conv_csv/mlem_{scale}_{mode}.csv"
    return f"results/{source}/conv_csv/{mode}_{scale}.csv"


def _hdf5_path(source, scale, mode):
    if source == PLATFORM_B:
        suffix = "" if scale == "256" else "_512"
        return f"results/{PLATFORM_B}/output_{mode.replace('-', '_')}{suffix}.hdf5"
    return f"results/{source}/{mode}_{scale}.hdf5"


def plot_osem_convergence(source=PLATFORM_B):
    if source != PLATFORM_B:
        print("OSEM sweep is Platform B/gpu-opt/256^3 only (Platform A has "
              "no OSEM data) -- skipping osem for --source", source)
        return
    configs = [1, 3, 5, 15, 25]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for s in configs:
        cum_t, logliks = _read_csv(f"results/{PLATFORM_B}/conv_csv/osem_s{s}.csv")
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


def plot_mlem_convergence(source=PLATFORM_B):
    suf = _suffix(source)
    hw_label = "Intel Core i7-5820K + AMD Hawaii PRO" if source == PLATFORM_A else "Intel Xeon E5-2620 + NVIDIA GeForce GTX 680"
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


def plot_slices(source=PLATFORM_B, scale="256"):
    suf = _suffix(source)
    hw_label = "Intel Core i7-5820K + AMD Hawaii PRO" if source == PLATFORM_A else "Intel Xeon E5-2620 + NVIDIA GeForce GTX 680"
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
    parser.add_argument("--source", default=PLATFORM_B, choices=[PLATFORM_B, PLATFORM_A],
                         help=f"which machine's data to plot (default: {PLATFORM_B})")
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
