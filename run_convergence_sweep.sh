#!/bin/bash
# Convergence sweeps with full per-epoch log-convergence CSVs, for plotting.
# Run on kale. Resumable: skips any config whose CSV already has EPOCHS rows.
set -e

DATA256=/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5
DATA512=/lgrp/edu-2026-1-gpulab/proj_512_75.hdf5
EPOCHS=100
mkdir -p conv_csv

# returns 0 (true) if the csv exists and its last row is epoch $EPOCHS
is_complete() {
  local csv="$1"
  [ -f "$csv" ] && [ "$(tail -1 "$csv" | cut -d, -f1)" = "$EPOCHS" ]
}

echo "=== OSEM sweep, 256^3, gpu-opt, S in {1,3,5,15,25} ==="
for S in 1 3 5 15 25; do
  csv="conv_csv/osem_s${S}.csv"
  if is_complete "$csv"; then
    echo "--- S=$S already complete, skipping ---"
    continue
  fi
  echo "--- S=$S ---"
  build/ct_recon --data "$DATA256" --out /tmp/osem_s${S}.hdf5 \
    --mode gpu-opt --epochs $EPOCHS --subsets $S --kernels kernels \
    --log-convergence "$csv"
done

echo ""
echo "=== Plain MLEM convergence, all four modes, 256^3 ==="
for MODE in cpu gpu-buf gpu-img gpu-opt; do
  csv="conv_csv/mlem_256_${MODE}.csv"
  if is_complete "$csv"; then
    echo "--- $MODE (256^3) already complete, skipping ---"
    continue
  fi
  echo "--- $MODE (256^3) ---"
  build/ct_recon --data "$DATA256" --out /tmp/mlem_256_${MODE}.hdf5 \
    --mode $MODE --epochs $EPOCHS --kernels kernels \
    --log-convergence "$csv"
done

echo ""
echo "=== Plain MLEM convergence, GPU modes, 512^3 (fast) ==="
for MODE in gpu-buf gpu-img gpu-opt; do
  csv="conv_csv/mlem_512_${MODE}.csv"
  if is_complete "$csv"; then
    echo "--- $MODE (512^3) already complete, skipping ---"
    continue
  fi
  echo "--- $MODE (512^3) ---"
  build/ct_recon --data "$DATA512" --out /tmp/mlem_512_${MODE}.hdf5 \
    --mode $MODE --epochs $EPOCHS --samples 512 --kernels kernels \
    --log-convergence "$csv"
done

echo ""
echo "=== Plain MLEM convergence, cpu, 512^3 (slow, ~1hr, OOM-risk on this shared machine) ==="
csv="conv_csv/mlem_512_cpu.csv"
if is_complete "$csv"; then
  echo "--- cpu (512^3) already complete, skipping ---"
else
  echo "--- cpu (512^3) ---"
  build/ct_recon --data "$DATA512" --out /tmp/mlem_512_cpu.hdf5 \
    --mode cpu --epochs $EPOCHS --samples 512 --kernels kernels \
    --log-convergence "$csv"
fi

echo ""
echo "=== done, CSVs in conv_csv/ ==="
ls -la conv_csv/
