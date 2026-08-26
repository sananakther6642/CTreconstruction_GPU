#!/bin/bash
# ARCHIVED: paths assume this script runs from repo root (its original
# location before the temp scripts/ cleanup move) -- "../python/X.py" and
# any "../*-worktree" paths need re-checking before rerunning from here.
# Full 100-epoch validation, 256^3 + 512^3, all four modes, on kale.
# For README hardware-numbers update.
set -e

echo "=== hardware ==="
hostname
nproc
grep "model name" /proc/cpuinfo | head -1

echo ""
echo "=== 256^3, 100 epochs, all four modes ==="
make run-cpu     EPOCHS=100
make run-gpu-buf EPOCHS=100
make run-gpu-img EPOCHS=100
make run-gpu-opt EPOCHS=100
python3 ../python/validate.py

echo ""
echo "=== OSEM S=5, 100 epochs (256^3), for the record ==="
build/ct_recon --data /lgrp/edu-2026-1-gpulab/proj_256_75.hdf5 --out output_osem_s5_100ep.hdf5 \
  --mode gpu-opt --epochs 100 --subsets 5 --kernels kernels

echo ""
echo "=== 512^3, 100 epochs, all four modes ==="
make run-cpu-512     EPOCHS=100
make run-gpu-buf-512 EPOCHS=100
make run-gpu-img-512 EPOCHS=100
make run-gpu-opt-512 EPOCHS=100
python3 ../python/validate.py 512

echo ""
echo "=== all done ==="
