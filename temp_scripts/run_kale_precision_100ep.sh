#!/bin/bash
# gpu-img/gpu-opt precision re-test at 100 epochs (kale, GTX 680).
#
# WHY THIS EXISTS
# Guangpu Yang's review set the bar at "mse at 10-8 level is good". Exactly
# one reported number misses it: gpu-img/gpu-opt at 256^3 = 1.1278e-07.
# (512^3 already passes at 1.232e-09; gpu-buf passes everywhere at
# 1.1477e-10 / 9.534e-11; OSEM's 1.061e-04 was explicitly excused as an
# "other update path".)
#
# The obvious fix -- CLK_FILTER_NEAREST + manual float32 blend replacing the
# hardware sampler's quantized weights -- was already built and measured as a
# NEGATIVE result on 2026-08-26 (branch `hybrid-precision`; see
# docs/correctness-history.md's hybrid-precision section). BUT that entire
# experiment ran at 10 EPOCHS, where its own baseline was 8.42e-10 -- already
# ~134x better than the 100-epoch number that actually fails. It measured the
# fix in a regime with essentially nothing to fix. This script re-runs it at
# the epoch count that matters.
#
# That 10ep -> 100ep growth (8.42e-10 -> 1.1278e-07, ~134x for 10x the
# iterations) is itself the second hypothesis under test here: error
# compounding through MLEM's `v *= bp_ratio/bp_ones`, whose only guard is
# `bp_ones > 1e-10f` (identical in src/ct_cpu.c and kernels/bp_buffer.cl), so
# FOV-edge voxels with near-zero normalizers amplify tiny differences every
# epoch. Arm C measures that directly and costs almost nothing on top.
#
# Run inside tmux (~20 min total):
#   tmux new -s precision
#   cd ~/CTreconstruction_GPU_pool15     # or wherever this repo is checked out
#   git checkout precision-mse-100ep && git pull
#   bash temp_scripts/run_kale_precision_100ep.sh 2>&1 | tee ct_precision_run.log
set -e
set -o pipefail   # otherwise `cmd | tee log` hides cmd's exit status
cd "$(dirname "$0")/.."

DATA256=/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5
EPOCHS=100
OUT=results_precision_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT/baseline" "$OUT/hybrid" "$OUT/curve" "$OUT/logs"

echo "=== $(date) : build ==="
make clean && make
ls -l build/ct_recon

run() {  # run <outdir> <tag> <mode> <epochs> [extra args...]
  local d="$1" tag="$2" mode="$3" ep="$4"; shift 4
  echo ""
  echo "--- $(date +%H:%M:%S)  $tag  (mode=$mode epochs=$ep) ---"
  build/ct_recon --data "$DATA256" --out "$d/output_${tag}.hdf5" \
    --mode "$mode" --epochs "$ep" --kernels kernels "$@" \
    2>&1 | tee "$OUT/logs/${tag}_$(basename "$d").log"
}

# ── Arm A: baseline, HYBRID off ────────────────────────────────────────
# Must reproduce the known 1.1278e-07 for gpu-img/gpu-opt. If it doesn't,
# something else changed and the rest of this run is not interpretable.
echo ""
echo "=== $(date) : ARM A -- baseline (HYBRID_PRECISION unset), 100 epochs ==="
unset HYBRID_PRECISION HYBRID_RADIUS_FRAC HYBRID_GRAD_THRESH
run "$OUT/baseline" cpu      cpu     $EPOCHS
run "$OUT/baseline" gpu_buf  gpu-buf $EPOCHS
run "$OUT/baseline" gpu_img  gpu-img $EPOCHS
run "$OUT/baseline" gpu_opt  gpu-opt $EPOCHS

echo ""
echo "=== ARM A validation ==="
python3 python/validate.py --dir "$OUT/baseline" 2>&1 | tee "$OUT/logs/validate_baseline.txt"

# ── Arm B: full-coverage manual blend ──────────────────────────────────
# radius_frac tiny + grad_thresh 0 => manual path everywhere, no gating.
# Reuses Arm A's cpu/gpu-buf outputs as the comparison reference (they are
# unaffected by HYBRID_PRECISION -- it only touches the image/opt programs).
echo ""
echo "=== $(date) : ARM B -- full-coverage manual blend, 100 epochs ==="
cp "$OUT/baseline/output_cpu.hdf5"     "$OUT/hybrid/" 2>/dev/null || true
cp "$OUT/baseline/output_gpu_buf.hdf5" "$OUT/hybrid/" 2>/dev/null || true
export HYBRID_PRECISION=1 HYBRID_RADIUS_FRAC=0.001 HYBRID_GRAD_THRESH=0
run "$OUT/hybrid" gpu_img gpu-img $EPOCHS
run "$OUT/hybrid" gpu_opt gpu-opt $EPOCHS
unset HYBRID_PRECISION HYBRID_RADIUS_FRAC HYBRID_GRAD_THRESH

echo ""
echo "=== ARM B validation ==="
python3 python/validate.py --dir "$OUT/hybrid" 2>&1 | tee "$OUT/logs/validate_hybrid.txt"

# ── Arm C: MSE-vs-epoch curve (amplification hypothesis) ───────────────
# Linear growth  => independent per-iteration error accumulating.
# Superlinear    => multiplicative amplification (the bp_ones/1e-10 path).
# 100ep point already exists in Arm A; this fills 10/25/50.
echo ""
echo "=== $(date) : ARM C -- MSE vs epoch count (10/25/50; 100 from Arm A) ==="
for EP in 10 25 50; do
  d="$OUT/curve/ep$EP"; mkdir -p "$d"
  run "$d" cpu     cpu     $EP
  run "$d" gpu_img gpu-img $EP
  echo "--- validate @ $EP epochs ---"
  python3 python/validate.py --dir "$d" 2>&1 | tee "$OUT/logs/validate_ep$EP.txt"
done

echo ""
echo "=== $(date) : DONE.  Results under $OUT/ ==="
echo ""
echo "READ THE RESULT LIKE THIS:"
echo "  1. Arm A gpu-img MSE should be ~1.128e-07 (reproduces the failing number)."
echo "     If it isn't, stop -- the baseline moved and nothing else is comparable."
echo "  2. Arm B gpu-img MSE <= 1e-08  => the sampler WAS the lever at 100 epochs,"
echo "     the 2026-08-26 negative result was an artifact of its 10-epoch scope,"
echo "     and this path should ship (gated). Note its speed cost in the same logs."
echo "  3. Arm B no better than Arm A => sampler is genuinely not the lever at any"
echo "     epoch count; the 10-epoch conclusion holds after all."
echo "  4. Arm C: plot MSE vs {10,25,50,100}. Superlinear growth points at"
echo "     multiplicative amplification via bp_ones, not per-sample precision --"
echo "     that would make an FOV mask / tighter bp_ones guard the real fix."
grep -H "gpu-img\|gpu-opt" "$OUT"/logs/validate_*.txt || true
