#!/bin/bash
# Correctness gate: bit-identical MSE check for gpu-buf.
# Diffs gpu-buf's output against a FIXED baseline commit, same data, same
# epoch count.
#
# IMPORTANT: BASELINE_SHA must be a fixed commit, not a branch name. This
# script is meant to be run FROM the `features` branch while optimizing
# it -- `git worktree add ... features` would check out whatever
# `features` currently points to, which after committing an optimization
# IS the current branch, silently comparing it to itself (always
# "bit-identical: True" with zero signal). Update BASELINE_SHA only when
# you deliberately want to move the baseline forward (e.g. after a batch
# of changes is confirmed correct and becomes the new reference point).
#
# Run on the NVIDIA GTX 680 (has the GPU/OpenCL runtime). Uses a worktree for the
# baseline checkout so it never touches this directory's live branch.
set -e

EPOCHS=${EPOCHS:-10}
DATA=/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5
BASELINE_WORKTREE="../gpu-buf-speed-baseline-worktree"
# Last commit before the bit-identical optimization pass (B1-B3 + OSEM
# step 1, see the plan) -- the pybind FTZ/DAZ + env-ordering fixes and
# the elementwise-loop parallelization, but nothing from this pass.
BASELINE_SHA="ae1a304"

echo "=== building gpu-buf-speed (current branch) ==="
make clean && make
build/ct_recon --data "$DATA" --out /tmp/out_new.hdf5 \
  --mode gpu-buf --epochs "$EPOCHS" --kernels kernels

echo ""
echo "=== building baseline ($BASELINE_SHA) in worktree ==="
if [ -d "$BASELINE_WORKTREE" ]; then
  current_sha=$(git -C "$BASELINE_WORKTREE" rev-parse HEAD)
  pinned_sha=$(git rev-parse "$BASELINE_SHA")
  if [ "$current_sha" != "$pinned_sha" ]; then
    echo "existing worktree is at $current_sha, expected $pinned_sha -- removing and recreating"
    git worktree remove "$BASELINE_WORKTREE" --force
  fi
fi
if [ ! -d "$BASELINE_WORKTREE" ]; then
  git worktree add "$BASELINE_WORKTREE" "$BASELINE_SHA"
fi
( cd "$BASELINE_WORKTREE" && make clean && make && \
  build/ct_recon --data "$DATA" --out /tmp/out_baseline.hdf5 \
    --mode gpu-buf --epochs "$EPOCHS" --kernels kernels )

echo ""
echo "=== diff ==="
python3 -c "
import h5py, numpy as np
a = h5py.File('/tmp/out_baseline.hdf5','r')['Volume'][:]
b = h5py.File('/tmp/out_new.hdf5','r')['Volume'][:]
if a.shape != b.shape:
    print('SHAPE MISMATCH', a.shape, b.shape); raise SystemExit(1)
diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
print('max abs diff:', diff.max())
print('bit-identical:', np.array_equal(a, b))
"

rm -f /tmp/out_new.hdf5 /tmp/out_baseline.hdf5
echo ""
echo "If bit-identical: True -> A1/A3 proven correctness-neutral, done."
echo "If False -> STOP, a bounds case was not actually redundant, investigate before shipping."
