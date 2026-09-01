#!/bin/bash
# FOV mask sweep (kale, GTX 680, 256^3, 100 epochs).
#
# WHY
# The gpu-img/gpu-opt 256^3 MSE of 1.128e-07 is NOT an interpolation
# problem. Measured 2026-09-01: bypassing the hardware sampler entirely
# (full-coverage manual float32 blend) moved it only 6% -> 1.061e-07, at a
# 55-58% speed cost. What actually drives it is MLEM's update
#   v *= bp_ratio / bp_ones
# guarded only by bp_ones > 1e-10f. bp_ones is the sensitivity map, so at
# FOV-edge voxels it is genuinely near zero and that division multiplies by
# up to ~1e10, amplifying any tiny difference every epoch. Evidence:
# MSE(gpu-img, gpu-buf) went 8.4e-10 -> 1.4e-8 -> 1.6e-7 across 10/25/50
# epochs (190x for 5x the iterations -- wildly superlinear), then FELL to
# 1.1e-7 at 100. Error that merely accumulated could not decrease; a few
# unstable edge voxels oscillating can. In Arm A's own output gpu-buf had
# MORE outlier voxels than gpu-img (1743 vs 1172) yet 1000x lower MSE --
# it was never about how many voxels disagree, only how far a few of them
# travel, and gpu-img's worst sits at radius 115.3/128, ~90% out to the edge.
#
# WHAT THIS SWEEPS
# FOV_MASK_REL = cutoff as a fraction of max(bp_ones). Voxels below it are
# outside the reliably-sampled region and get zeroed (not frozen -- the
# volume starts at 1.0, so freezing would leave a bright rim). The same env
# var drives CPU and every GPU mode through one shared helper
# (src/utils.c fov_mask_threshold), so they cannot silently disagree.
#
# FOV_MASK_REL unset reproduces the legacy behaviour exactly (threshold
# 1e-10f, freeze-not-zero), which is what the rel=0 arm below confirms.
#
# NOTE: this CHANGES RECONSTRUCTION OUTPUT for every mode including the CPU
# reference. That is intended -- masking unsampled voxels is standard CT
# practice -- but it means whichever value is adopted requires re-running
# the reported benchmarks. Pick the value here, then re-run.
#
# Needs a CPU volume per setting: unlike the sampler test, the mask changes
# CPU output too, so a fixed reference cannot be reused across arms.
#
# Run (~35 min: 5 settings x (cpu 7min + 3 gpu modes ~1min)):
#   cd ~/ct-precision && git pull
#   tmux new -s fovmask
#   bash temp_scripts/run_kale_fovmask_sweep.sh 2>&1 | tee ct_fovmask_run.log
set -e
set -o pipefail
cd "$(dirname "$0")/.."

DATA256=/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5
EPOCHS=100
OUT=results_fovmask_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT/logs"

echo "=== $(date) : build ==="
make clean && make

# rel=0 is the control: must reproduce today's 1.128e-07 exactly.
for REL in 0 1e-4 1e-3 1e-2 5e-2; do
  d="$OUT/rel_$REL"; mkdir -p "$d"
  echo ""
  echo "############ $(date +%H:%M:%S)  FOV_MASK_REL=$REL ############"
  if [ "$REL" = "0" ]; then unset FOV_MASK_REL; else export FOV_MASK_REL=$REL; fi

  for M in cpu gpu-buf gpu-img gpu-opt; do
    tag=$(echo "$M" | tr '-' '_')
    echo "--- $(date +%H:%M:%S) $M @ rel=$REL ---"
    build/ct_recon --data "$DATA256" --out "$d/output_${tag}.hdf5" \
      --mode "$M" --epochs "$EPOCHS" --kernels kernels \
      2>&1 | tee "$OUT/logs/${tag}_rel${REL}.log"
  done

  echo "--- validation @ rel=$REL ---"
  python3 python/validate.py --dir "$d" 2>&1 | tee "$OUT/logs/validate_rel${REL}.txt"
  # Volumes are 67MB each; MSE is already captured above.
  rm -f "$d"/output_*.hdf5
done
unset FOV_MASK_REL

echo ""
echo "=== $(date) : SUMMARY ==="
for f in "$OUT"/logs/validate_rel*.txt; do
  echo "--- $(basename "$f") ---"
  grep -E "gpu-buf|gpu-img|gpu-opt" "$f" | grep MSE || true
done

echo ""
echo "HOW TO READ IT:"
echo "  rel=0 must reproduce gpu-img MSE ~1.128e-07. If not, stop."
echo "  Then pick the SMALLEST rel that brings gpu-img/gpu-opt <= 1e-08."
echo "  Watch gpu-buf too: it should stay ~1e-10 and must not degrade."
echo "  Also sanity-check 'max' per mode -- a mask that is too aggressive"
echo "  starts eating real signal, which shows up as a falling max value."
