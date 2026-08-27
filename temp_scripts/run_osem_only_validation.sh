#!/bin/bash
# ARCHIVED: paths assume this script runs from repo root (its original
# location before the temp_scripts/ cleanup move) -- "../python/X.py" and
# any "../*-worktree" paths need re-checking before rerunning from here.
# Validate perf-v2-osem-only: baseline + OSEM only, C2/C3/C5 removed.
# Checks: (1) all four modes still build and agree at default settings
# (byte-for-byte same as perf-v2/features since C2/C3/C5 were opt-in
# flags only), (2) OSEM still works correctly post-strip.
set -e

DATA=/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5
BIN=build/ct_recon
KDIR=kernels

echo "=== build ==="
make clean && make

echo ""
echo "=== 1. baseline regression: all four modes, default settings, 10 epochs ==="
make run-cpu EPOCHS=10
make run-gpu-buf EPOCHS=10
make run-gpu-img EPOCHS=10
make run-gpu-opt EPOCHS=10
python3 ../python/validate.py

echo ""
echo "=== 2. OSEM still works: --subsets 1 (should match default exactly) ==="
$BIN --data $DATA --out output_osem_s1_check.hdf5 \
    --mode gpu-opt --epochs 10 --subsets 1 --kernels $KDIR

echo ""
echo "=== 3. OSEM S sweep smoke test (S=1,5,15), 20 epochs, with convergence log ==="
for S in 1 5 15; do
    echo "--- S=$S ---"
    $BIN --data $DATA --out output_osem_only_S${S}.hdf5 \
        --mode gpu-opt --epochs 20 --subsets $S \
        --log-convergence conv_osem_only_S${S}.csv --kernels $KDIR
done

echo ""
echo "=== 4. compare S=1 vs S=5 vs S=15 output stats ==="
python3 -c "
import h5py, numpy as np
for S in [1,5,15]:
    v = h5py.File(f'output_osem_only_S{S}.hdf5','r')['Volume'][:]
    print(f'S={S:2d}  min={v.min():.4f} max={v.max():.4f} mean={v.mean():.6f} nan={np.isnan(v).sum()} inf={np.isinf(v).sum()}')
"

echo ""
echo "=== 5. confirm removed flags are actually gone (should all fail/error) ==="
set +e
$BIN --data $DATA --out /tmp/x.hdf5 --mode gpu-opt --epochs 1 --beta 100 --kernels $KDIR 2>&1 | head -3
echo "(--beta above should be an unrecognized-flag error, not silently accepted)"
$BIN --data $DATA --out /tmp/x.hdf5 --mode fdk --kernels $KDIR 2>&1 | head -3
echo "(--mode fdk above should fail, mode no longer exists)"
set -e

echo ""
echo "=== all checks done ==="
