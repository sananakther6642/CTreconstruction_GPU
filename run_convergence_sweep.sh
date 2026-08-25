#!/bin/bash
# OSEM convergence sweep with full per-epoch log-convergence CSVs, for plotting.
# Run on kale. 256^3, gpu-opt, 100 epochs, S in {1,3,5,15,25}.
set -e

DATA=/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5
mkdir -p conv_csv

for S in 1 3 5 15 25; do
  echo "=== S=$S ==="
  build/ct_recon --data "$DATA" --out /tmp/osem_s${S}.hdf5 \
    --mode gpu-opt --epochs 100 --subsets $S --kernels kernels \
    --log-convergence conv_csv/osem_s${S}.csv
done

echo "=== done, CSVs in conv_csv/ ==="
ls -la conv_csv/
