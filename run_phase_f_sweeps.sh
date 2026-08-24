#!/bin/bash
# Phase F: report-quality sweeps (S, FDK x OSEM 2x2, beta), 256^3, on kale.
set -e

DATA=/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5
BIN=build/ct_recon
KDIR=kernels

echo "=== S sweep: S in {1,3,5,15,25}, 100 epochs ==="
for S in 1 3 5 15 25; do
    echo "--- S=$S ---"
    $BIN --data $DATA --out output_S${S}_100ep.hdf5 \
        --mode gpu-opt --epochs 100 --subsets $S \
        --log-convergence conv_S${S}_100ep.csv --kernels $KDIR
done

echo "=== FDK x OSEM 2x2, 50 epochs ==="
$BIN --data $DATA --out output_2x2_ones_mlem.hdf5 \
    --mode gpu-opt --epochs 50 --subsets 1 --init ones \
    --log-convergence conv_2x2_ones_mlem.csv --kernels $KDIR

$BIN --data $DATA --out output_2x2_fdk_mlem.hdf5 \
    --mode gpu-opt --epochs 50 --subsets 1 --init fdk \
    --log-convergence conv_2x2_fdk_mlem.csv --kernels $KDIR

$BIN --data $DATA --out output_2x2_ones_osem.hdf5 \
    --mode gpu-opt --epochs 50 --subsets 5 --init ones \
    --log-convergence conv_2x2_ones_osem.csv --kernels $KDIR

$BIN --data $DATA --out output_2x2_fdk_osem.hdf5 \
    --mode gpu-opt --epochs 50 --subsets 5 --init fdk \
    --log-convergence conv_2x2_fdk_osem.csv --kernels $KDIR

echo "=== beta sweep: beta in {0,100,1000,5000}, 100 epochs ==="
for B in 0 100 1000 5000; do
    echo "--- beta=$B ---"
    $BIN --data $DATA --out output_beta${B}_100ep.hdf5 \
        --mode gpu-opt --epochs 100 --subsets 1 --beta $B \
        --log-convergence conv_beta${B}_100ep.csv --kernels $KDIR
done

echo "=== all sweeps done ==="
echo "CSVs:"
ls -la conv_S*_100ep.csv conv_2x2_*.csv conv_beta*_100ep.csv
