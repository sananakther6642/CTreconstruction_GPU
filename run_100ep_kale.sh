#!/bin/bash
# Full 100-epoch validation across all four modes, 256^3, on kale.
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
python3 validate.py

echo ""
echo "=== OSEM S=5, 100 epochs, for the record ==="
build/ct_recon --data /lgrp/edu-2026-1-gpulab/proj_256_75.hdf5 --out output_osem_s5_100ep.hdf5 \
  --mode gpu-opt --epochs 100 --subsets 5 --kernels kernels

echo ""
echo "=== all done ==="
