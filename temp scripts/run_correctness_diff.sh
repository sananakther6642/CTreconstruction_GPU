#!/bin/bash
# gpu-buf-speed A1/A3 correctness gate: bit-identical MSE check.
# A1 removed redundant per-tap bounds checks in trilinear_buf, A3 hoisted
# Nxz*Ny. Both are argued as pure no-op removals -- this proves it by
# diffing gpu-buf's output against the pre-change `features` baseline,
# same data, same epoch count.
#
# Run on the NVIDIA GTX 680 (has the GPU/OpenCL runtime). Uses a worktree for the
# baseline checkout so it never touches this directory's live branch.
set -e

EPOCHS=${EPOCHS:-10}
DATA=/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5
BASELINE_WORKTREE="../gpu-buf-speed-baseline-worktree"

echo "=== building gpu-buf-speed (current branch) ==="
make clean && make
build/ct_recon --data "$DATA" --out /tmp/out_new.hdf5 \
  --mode gpu-buf --epochs "$EPOCHS" --kernels kernels

echo ""
echo "=== building features baseline (pre-A1/A2/A3) in worktree ==="
if [ ! -d "$BASELINE_WORKTREE" ]; then
  git worktree add "$BASELINE_WORKTREE" features
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
