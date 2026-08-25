#!/bin/bash
# Full pool15-01 (AMD Hawaii PRO) capture: speed-table timings + convergence
# CSVs, all four modes, both dataset sizes, 100 epochs. Resumable -- safe to
# rerun; skips anything already complete.
#
# Run inside tmux so it survives a disconnect:
#   tmux new -s pool15
#   ./run_pool15_full.sh
#   (detach: Ctrl+B then D)
# Reattach later: tmux attach -t pool15
#
# All output lands under pool15/ so it's easy to scp/rsync back as one unit.
set -e

DATA256=/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5
DATA512=/lgrp/edu-2026-1-gpulab/proj_512_75.hdf5
EPOCHS=100
OUT=pool15
mkdir -p "$OUT/hdf5" "$OUT/conv_csv" "$OUT/logs"

is_complete() {
  local csv="$1"
  [ -f "$csv" ] && [ "$(tail -1 "$csv" | cut -d, -f1)" = "$EPOCHS" ]
}

echo "=== hardware ===" | tee "$OUT/hardware.txt"
hostname | tee -a "$OUT/hardware.txt"
nproc | tee -a "$OUT/hardware.txt"
grep "model name" /proc/cpuinfo | head -1 | tee -a "$OUT/hardware.txt"
clinfo 2>/dev/null | grep -i "device name" | head -1 | tee -a "$OUT/hardware.txt"

run_one() {
  local mode="$1" scale="$2" data="$3" extra="$4"
  local tag="${mode}_${scale}"
  local csv="$OUT/conv_csv/${tag}.csv"
  local hdf5="$OUT/hdf5/${tag}.hdf5"
  local log="$OUT/logs/${tag}.log"

  if is_complete "$csv"; then
    echo "--- $tag already complete, skipping ---"
    return
  fi
  echo "--- $tag ---"
  build/ct_recon --data "$data" --out "$hdf5" \
    --mode "$mode" --epochs $EPOCHS $extra --kernels kernels \
    --log-convergence "$csv" 2>&1 | tee "$log"
}

echo ""
echo "=== 256^3, all four modes ==="
for MODE in cpu gpu-buf gpu-img gpu-opt; do
  run_one "$MODE" 256 "$DATA256" ""
done

echo ""
echo "=== 512^3, GPU modes (fast, run before cpu) ==="
for MODE in gpu-buf gpu-img gpu-opt; do
  run_one "$MODE" 512 "$DATA512" "--samples 512"
done

echo ""
echo "=== 512^3, cpu (slow, ~1hr) ==="
run_one cpu 512 "$DATA512" "--samples 512"

echo ""
echo "=== validation (copies into validate.py's expected names first) ==="
cp "$OUT/hdf5/cpu_256.hdf5"     output_cpu.hdf5         2>/dev/null || true
cp "$OUT/hdf5/gpu-buf_256.hdf5" output_gpu_buf.hdf5     2>/dev/null || true
cp "$OUT/hdf5/gpu-img_256.hdf5" output_gpu_img.hdf5     2>/dev/null || true
cp "$OUT/hdf5/gpu-opt_256.hdf5" output_gpu_opt.hdf5     2>/dev/null || true
cp "$OUT/hdf5/cpu_512.hdf5"     output_cpu_512.hdf5     2>/dev/null || true
cp "$OUT/hdf5/gpu-buf_512.hdf5" output_gpu_buf_512.hdf5 2>/dev/null || true
cp "$OUT/hdf5/gpu-img_512.hdf5" output_gpu_img_512.hdf5 2>/dev/null || true
cp "$OUT/hdf5/gpu-opt_512.hdf5" output_gpu_opt_512.hdf5 2>/dev/null || true
python3 validate.py     2>&1 | tee "$OUT/validate_256.txt" || true
python3 validate.py 512 2>&1 | tee "$OUT/validate_512.txt" || true

echo ""
echo "=== done. Everything under $OUT/ ==="
find "$OUT" -type f | sort
