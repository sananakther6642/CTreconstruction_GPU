#!/usr/bin/env python3
"""Phase 1 attribution: how much of gpu-img/gpu-opt's MSE-vs-CPU comes from
fp (forward projection) vs bp (backprojection)?

fp_image.cl does ~n_samples 8-tap hardware-sampled 3D reads per ray;
bp_image.cl/bp_buffer_opt.cl do one 4-tap hardware-sampled 2D read per
angle (75). fp is the likely dominant error source by read-count alone --
this measures it instead of assuming it.

Compares gpu-buf (exact, no hardware sampler) against gpu-img (hardware
sampler) for fp and bp in isolation, using the six .hdf5 files produced by:
    make run-op-fp run-op-bp run-op-fp-gpubuf run-op-fp-gpuimg \
        run-op-bp-gpubuf run-op-bp-gpuimg

fp's outputs are projections (fp_cpu.hdf5 stores a 'Projection' or is
loaded via save_hdf5_proj -- check the actual key on first run); bp's
outputs are volumes ('Volume', same as every other output).
"""
import sys
import numpy as np
import h5py


def load(path, key_candidates=('Volume', 'Projection', 'proj')):
    with h5py.File(path, 'r') as f:
        for k in key_candidates:
            if k in f:
                return f[k][:]
        # fall back to whatever the single dataset is
        keys = list(f.keys())
        if len(keys) == 1:
            return f[keys[0]][:]
        raise KeyError(f"{path}: no known key found, has {keys}")


def report(label, ref, test):
    ref = ref.astype(np.float64)
    test = test.astype(np.float64)
    if ref.shape != test.shape:
        print(f"{label}: SHAPE MISMATCH {ref.shape} vs {test.shape}")
        return None
    diff = np.abs(ref - test)
    mse = float(np.mean(diff ** 2))
    rms = np.sqrt(mse)
    rng = float(ref.max() - ref.min())
    rel = 100.0 * rms / rng if rng > 0 else float('nan')
    maxdiff = float(diff.max())
    print(f"{label:<20} MSE={mse:.6e}  RMS={rms:.6e}  range={rng:.4f}  "
          f"RMS%={rel:.6f}%  maxdiff={maxdiff:.6e}")
    return rms


print("=== fp: gpu-buf (exact) vs gpu-img (hardware sampler) ===")
fp_buf = load('fp_gpubuf.hdf5')
fp_img = load('fp_gpuimg.hdf5')
fp_rms = report('fp gpu-img vs gpu-buf', fp_buf, fp_img)

print("\n=== bp: gpu-buf (exact) vs gpu-img (hardware sampler) ===")
bp_buf = load('bp_gpubuf.hdf5')
bp_img = load('bp_gpuimg.hdf5')
bp_rms = report('bp gpu-img vs gpu-buf', bp_buf, bp_img)

print("\n=== sanity: gpu-buf vs cpu (should be ~exact, both manual interp) ===")
fp_cpu = load('fp_cpu.hdf5')
bp_cpu = load('bp_cpu.hdf5')
report('fp gpu-buf vs cpu', fp_cpu, fp_buf)
report('bp gpu-buf vs cpu', bp_cpu, bp_buf)

print("\n=== ATTRIBUTION ===")
if fp_rms is not None and bp_rms is not None:
    if bp_rms == 0:
        ratio = float('inf')
    else:
        ratio = fp_rms / bp_rms
    print(f"fp RMS = {fp_rms:.6e}")
    print(f"bp RMS = {bp_rms:.6e}")
    print(f"fp/bp ratio = {ratio:.2f}x")
    if ratio > 3:
        print("=> fp DOMINATES. Phase 2 should target fp_image.cl only.")
    elif ratio < 0.33:
        print("=> bp DOMINATES. Phase 2 should target bp_image.cl + bp_buffer_opt.cl.")
    else:
        print("=> COMPARABLE. Phase 2 would need all three kernels -- reconsider.")
