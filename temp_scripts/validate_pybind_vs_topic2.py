#!/usr/bin/env python3
"""
Compares pybind's reconstruction output(s) against Topic_2's Python
reference output. Mirrors python/validate.py's MSE-vs-Python-Ref
column, but validate.py only knows about the CLI's own output_*.hdf5
filenames -- this targets run_kale_full_100ep.sh's pybind_output_*.hdf5
files specifically, since neither script knew about the other's
naming when written.

Usage (from the repo root, after both run_kale_full_100ep.sh and
run_topic2_20ep.sh have produced their outputs):

    python3 temp_scripts/validate_pybind_vs_topic2.py \
        --results-dir results_kalefull_100ep_YYYYMMDD_HHMMSS \
        --topic2-output output_python_reconstruction.hdf5

--topic2-output defaults to output_python_reconstruction.hdf5 (256^3
only -- run_topic2_20ep.sh runs Topic_2_CTreconstruction.py, which is
hardcoded to the 256^3 dataset; there is no 512^3 Python reference,
confirmed infeasible on both machines this session's own history
checked -- see README.md's "512^3 not possible with Python reference"
note). Only 256^3 pybind outputs are compared for that reason; 512^3
pybind outputs in --results-dir are skipped with a note, not silently
ignored.
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np


def load_volume(path):
    with h5py.File(path, "r") as f:
        return f["Volume"][:].astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True,
                     help="the results_kalefull_100ep_* directory from run_kale_full_100ep.sh")
    ap.add_argument("--topic2-output", default="output_python_reconstruction.hdf5",
                     help="Topic_2's own output file (default: repo-root-relative)")
    args = ap.parse_args()

    if not os.path.isfile(args.topic2_output):
        print(f"ERROR: Topic_2 output not found at {args.topic2_output}")
        print("  Has run_topic2_20ep.sh finished? Check run_topic2_20ep.DONE.")
        sys.exit(1)

    ref = load_volume(args.topic2_output)
    print(f"Topic_2 reference: {args.topic2_output}  shape={ref.shape}")
    print()

    pybind_files = sorted(glob.glob(os.path.join(args.results_dir, "pybind_output_*_256.hdf5")))
    skipped_512 = sorted(glob.glob(os.path.join(args.results_dir, "pybind_output_*_512.hdf5")))

    if not pybind_files:
        print(f"No pybind_output_*_256.hdf5 files found in {args.results_dir}")
        sys.exit(1)

    print(f"{'mode':<28} {'shape match':<12} {'MSE vs Topic_2':<16} {'max abs diff':<14} {'nan':<5} {'inf':<5}")
    print("-" * 90)
    for path in pybind_files:
        name = os.path.basename(path).replace("pybind_output_", "").replace("_256.hdf5", "")
        try:
            v = load_volume(path)
        except Exception as e:
            print(f"{name:<28} FAILED TO LOAD: {e}")
            continue
        shape_ok = v.shape == ref.shape
        n_nan = int(np.isnan(v).sum())
        n_inf = int(np.isinf(v).sum())
        if shape_ok:
            diff = np.abs(v - ref)
            mse = float(np.mean(diff ** 2))
            maxdiff = float(np.max(diff))
            print(f"{name:<28} {'yes':<12} {mse:<16.4e} {maxdiff:<14.4f} {n_nan:<5} {n_inf:<5}")
        else:
            print(f"{name:<28} {'NO ('+str(v.shape)+')':<12} {'--':<16} {'--':<14} {n_nan:<5} {n_inf:<5}")

    if skipped_512:
        print()
        print(f"Skipped {len(skipped_512)} 512^3 pybind output(s) -- no 512^3 Python reference exists")
        print("(confirmed infeasible on both machines, out-of-memory; see README.md).")
        for p in skipped_512:
            print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
