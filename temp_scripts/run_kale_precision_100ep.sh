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
# fix in a regime with essentially nothing left to fix. This script re-runs it
# at the epoch count that matters.
#
# That 10ep -> 100ep growth (8.42e-10 -> 1.1278e-07, ~134x for 10x the
# iterations) is itself the second hypothesis under test: error compounding
# through MLEM's `v *= bp_ratio/bp_ones`, whose only guard is
# `bp_ones > 1e-10f` (identical in src/ct_cpu.c and kernels/bp_buffer.cl), so
# FOV-edge voxels with near-zero normalizers amplify tiny differences every
# epoch. Arm C measures that directly.
#
# NO CPU RUNS. cpu at 256^3 is ~4.2 s/epoch (~10x gpu-buf) and would have
# dominated this script's runtime, for two avoidable reasons:
#   - Arms A/B need a CPU volume only as the comparison REFERENCE, so a
#     pre-existing 100-epoch one is supplied via PRECISION_CPU_REF (see below).
#   - Arm C does not need cpu at all: gpu-buf IS the manual float32
#     interpolation path, so MSE(gpu-img, gpu-buf) isolates the hardware
#     sampler's own contribution with the CPU/GPU toolchain difference
#     divided out on both sides.
# Total runtime is therefore ~5 min rather than ~20.
#
# REQUIRED: a 100-epoch 256^3 CPU volume to score against.
#   scp submission_outputs/gtx680/output_cpu.hdf5 kale:~/cpu_ref_256.hdf5
#   export PRECISION_CPU_REF=~/cpu_ref_256.hdf5
# This reference is self-validating: the 1.1278e-07 figure in the README was
# produced against this exact volume, so if Arm A reproduces ~1.128e-07 the
# reference is confirmed good. If Arm A comes out somewhere else, suspect the
# reference FIRST, before reading anything into Arms B/C.
#
# Run inside tmux, in its own clone -- never in a checkout with a Hawaii
# sweep running (this does `make clean`, and ct_recon reads kernels/ at
# RUNTIME via --kernels):
#   cd ~ && git clone -b precision-mse-100ep \
#       https://github.com/sananakther6642/CTreconstruction_GPU.git ct-precision
#   tmux new -s precision
#   cd ~/ct-precision
#   export PRECISION_CPU_REF=~/cpu_ref_256.hdf5
#   bash temp_scripts/run_kale_precision_100ep.sh 2>&1 | tee ct_precision_run.log
#
# Disk: ~350MB peak. Safe alongside the Hawaii sweep on a shared NFS home.
set -e
set -o pipefail   # otherwise `cmd | tee log` hides cmd's exit status
cd "$(dirname "$0")/.."

# Refuse to run inside a checkout that is mid-Hawaii-sweep: this script does
# `make clean` (deleting the binary that run still calls) and expects its own
# kernels/ (which ct_recon reads at RUNTIME via --kernels).
if [ -f ct_hawaiifull_run.log ] && [ ! -f ct_hawaiifull_DONE ]; then
  echo "ERROR: ct_hawaiifull_run.log exists without ct_hawaiifull_DONE --"
  echo "  a Hawaii sweep looks unfinished in this directory. Running here"
  echo "  would 'make clean' the binary it still needs and swap its kernels/."
  echo "  Use a separate clone (see this script's header)."
  exit 1
fi

if [ -z "$PRECISION_CPU_REF" ] || [ ! -f "$PRECISION_CPU_REF" ]; then
  echo "ERROR: PRECISION_CPU_REF must point at a 100-epoch 256^3 CPU volume."
  echo "  From the local checkout:"
  echo "    scp submission_outputs/gtx680/output_cpu.hdf5 kale:~/cpu_ref_256.hdf5"
  echo "  Then on kale:"
  echo "    export PRECISION_CPU_REF=~/cpu_ref_256.hdf5"
  echo "  (Arms A/B score against it; Arm C does not need it.)"
  exit 1
fi

DATA256=/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5
EPOCHS=100
OUT=results_precision_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT/baseline" "$OUT/hybrid" "$OUT/curve" "$OUT/logs"

CPU_REF_ABS="$(cd "$(dirname "$PRECISION_CPU_REF")" && pwd)/$(basename "$PRECISION_CPU_REF")"
echo "CPU reference: $CPU_REF_ABS ($(du -h "$CPU_REF_ABS" | cut -f1))"

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
# stop -- either the CPU reference is wrong or the baseline moved, and
# nothing in Arms B/C is interpretable until that is resolved.
echo ""
echo "=== $(date) : ARM A -- baseline (HYBRID_PRECISION unset), 100 epochs ==="
unset HYBRID_PRECISION HYBRID_RADIUS_FRAC HYBRID_GRAD_THRESH
ln -sf "$CPU_REF_ABS" "$OUT/baseline/output_cpu.hdf5"
run "$OUT/baseline" gpu_buf  gpu-buf $EPOCHS
run "$OUT/baseline" gpu_img  gpu-img $EPOCHS
run "$OUT/baseline" gpu_opt  gpu-opt $EPOCHS

echo ""
echo "=== ARM A validation ==="
python3 python/validate.py --dir "$OUT/baseline" 2>&1 | tee "$OUT/logs/validate_baseline.txt"

# ── Arm B: full-coverage manual blend ──────────────────────────────────
# radius_frac tiny + grad_thresh 0 => manual path everywhere, no gating.
# Reuses Arm A's cpu/gpu-buf as reference: HYBRID_PRECISION only touches the
# image and opt programs, so both are bit-identical across arms by
# construction. Symlinked, not copied (67MB each).
echo ""
echo "=== $(date) : ARM B -- full-coverage manual blend, 100 epochs ==="
ln -sf "$CPU_REF_ABS" "$OUT/hybrid/output_cpu.hdf5"
ln -sf "$PWD/$OUT/baseline/output_gpu_buf.hdf5" "$OUT/hybrid/output_gpu_buf.hdf5"
export HYBRID_PRECISION=1 HYBRID_RADIUS_FRAC=0.001 HYBRID_GRAD_THRESH=0
run "$OUT/hybrid" gpu_img gpu-img $EPOCHS
run "$OUT/hybrid" gpu_opt gpu-opt $EPOCHS
unset HYBRID_PRECISION HYBRID_RADIUS_FRAC HYBRID_GRAD_THRESH

echo ""
echo "=== ARM B validation ==="
python3 python/validate.py --dir "$OUT/hybrid" 2>&1 | tee "$OUT/logs/validate_hybrid.txt"

# ── Arm C: sampler error vs epoch count ────────────────────────────────
# Reference is gpu-buf, not cpu -- see the header. Both are GPU-fast, so the
# whole curve costs well under a minute.
echo ""
echo "=== $(date) : ARM C -- sampler error vs epoch count (gpu-buf reference) ==="
: > "$OUT/logs/curve.txt"
for EP in 10 25 50 100; do
  d="$OUT/curve/ep$EP"; mkdir -p "$d"
  run "$d" gpu_buf gpu-buf $EP
  run "$d" gpu_img gpu-img $EP
  python3 python/mse_pair.py "$d/output_gpu_img.hdf5" "$d/output_gpu_buf.hdf5" \
    "epochs=$EP" 2>&1 | tee -a "$OUT/logs/curve.txt"
  # MSE recorded above; volumes are 67MB each and nothing downstream reads
  # them. Drop them so this arm's footprint stays flat.
  rm -f "$d"/output_*.hdf5
done

echo ""
echo "=== ARM C curve ==="
cat "$OUT/logs/curve.txt"

echo ""
echo "=== $(date) : DONE.  Results under $OUT/ ==="
echo ""
echo "READ THE RESULT LIKE THIS:"
echo "  1. Arm A gpu-img MSE should be ~1.128e-07. If it isn't, stop -- suspect"
echo "     the CPU reference first; nothing else is comparable until it matches."
echo "  2. Arm B gpu-img MSE <= 1e-08  => the sampler WAS the lever at 100"
echo "     epochs, the 2026-08-26 negative result was an artifact of its"
echo "     10-epoch scope, and this path should ship (gated). Its speed cost is"
echo "     in the same logs -- the prior run measured 37-53% slower."
echo "  3. Arm B no better than Arm A => the sampler is genuinely not the lever"
echo "     at any epoch count, and the 10-epoch conclusion holds after all."
echo "  4. Arm C curve (gpu-img vs gpu-buf at 10/25/50/100 epochs):"
echo "     roughly LINEAR in epochs => per-sample sampler error just piling up,"
echo "     so the sampler is the whole story and Arm B is the fix."
echo "     SUPERLINEAR => multiplicative amplification through bp_ones, which no"
echo "     interpolation change can fix; the lever would be an FOV mask or a"
echo "     tighter bp_ones guard than today's 1e-10f. That changes CPU output"
echo "     too, so it needs an explicit go-ahead before implementing."
grep -H "gpu-img\|gpu-opt" "$OUT"/logs/validate_*.txt || true
