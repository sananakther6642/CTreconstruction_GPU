#!/bin/bash
# ARCHIVED: paths assume this script runs from repo root (its original
# location before the temp_scripts/ cleanup move) -- "../python/X.py" and
# any "../*-worktree" paths need re-checking before rerunning from here.
# Retry the 4 configs that failed on AMD Hawaii PRO's main run with
# "OpenCL error -1 at clGetDeviceIDs" (transient GPU unavailability,
# confirmed recovered via clinfo). Runs standalone, safe alongside the
# main run_hawaii_full.sh session since that's on Topic_2 (CPU-only) by
# the time this is needed. Same run_one/checkpoint logic as the main
# script, scoped to just these 4 -- commits+pushes after each one
# finishes, no manual intervention between configs.
#
# Run inside tmux so it survives a disconnect:
#   tmux new -s pool15-retry
#   ./retry_hawaii_gpu_configs.sh
#   (detach: Ctrl+B then D)
set -e

DATA256=/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5
DATA512=/lgrp/edu-2026-1-gpulab/proj_512_75.hdf5
EPOCHS=100
OUT=pool15
mkdir -p "$OUT/hdf5" "$OUT/conv_csv" "$OUT/logs"

WORKTREE_DIR="../pool15-results-worktree"

checkpoint() {
  local label="$1"
  local repo_root
  repo_root=$(git rev-parse --show-toplevel)
  (
    set +e
    if [ ! -d "$WORKTREE_DIR" ]; then
      if git show-ref --verify --quiet refs/heads/pool15-results; then
        git worktree add "$WORKTREE_DIR" pool15-results
      elif git ls-remote --exit-code --heads origin pool15-results >/dev/null 2>&1; then
        git fetch origin pool15-results
        git worktree add -B pool15-results "$WORKTREE_DIR" origin/pool15-results
      else
        git worktree add -B pool15-results "$WORKTREE_DIR"
      fi
    fi
    rsync -a --delete "$repo_root/$OUT/" "$WORKTREE_DIR/$OUT/"
    cd "$WORKTREE_DIR" || exit 1
    git add -f "$OUT"/logs 2>/dev/null
    git add -f "$OUT"/conv_csv 2>/dev/null
    if git diff --cached --quiet; then
      echo "checkpoint ($label): nothing new to commit"
    else
      git commit -m "pool15 checkpoint: retry $label ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
      git push -u origin pool15-results
    fi
  ) || echo "WARNING: checkpoint ($label) commit/push failed -- results still on disk under $OUT/, just not in git."
}

run_one() {
  local mode="$1" scale="$2" data="$3" extra="$4"
  local tag="${mode}_${scale}"
  local csv="$OUT/conv_csv/${tag}.csv"
  local hdf5="$OUT/hdf5/${tag}.hdf5"
  local log="$OUT/logs/${tag}.log"

  echo "--- retry: $tag ---"
  build/ct_recon --data "$data" --out "$hdf5" \
    --mode "$mode" --epochs $EPOCHS $extra --kernels kernels \
    --log-convergence "$csv" 2>&1 | tee "$log"

  if grep -q "OpenCL error" "$log"; then
    echo "!!! $tag still hit an OpenCL error -- GPU issue not resolved. Stopping here."
    exit 1
  fi
}

run_one gpu-opt 256 "$DATA256" ""
checkpoint "gpu-opt_256"

run_one gpu-buf 512 "$DATA512" "--samples 512"
checkpoint "gpu-buf_512"

run_one gpu-img 512 "$DATA512" "--samples 512"
checkpoint "gpu-img_512"

run_one gpu-opt 512 "$DATA512" "--samples 512"
checkpoint "gpu-opt_512"

echo ""
echo "=== all 4 configs done. regenerating validation + plots with full data ==="
cp "$OUT/hdf5/cpu_256.hdf5"     output_cpu.hdf5         2>/dev/null || true
cp "$OUT/hdf5/gpu-buf_256.hdf5" output_gpu_buf.hdf5     2>/dev/null || true
cp "$OUT/hdf5/gpu-img_256.hdf5" output_gpu_img.hdf5     2>/dev/null || true
cp "$OUT/hdf5/gpu-opt_256.hdf5" output_gpu_opt.hdf5     2>/dev/null || true
cp "$OUT/hdf5/cpu_512.hdf5"     output_cpu_512.hdf5     2>/dev/null || true
cp "$OUT/hdf5/gpu-buf_512.hdf5" output_gpu_buf_512.hdf5 2>/dev/null || true
cp "$OUT/hdf5/gpu-img_512.hdf5" output_gpu_img_512.hdf5 2>/dev/null || true
cp "$OUT/hdf5/gpu-opt_512.hdf5" output_gpu_opt_512.hdf5 2>/dev/null || true
python3 ../python/validate.py     2>&1 | tee "$OUT/validate_256.txt" || true
python3 ../python/validate.py 512 2>&1 | tee "$OUT/validate_512.txt" || true
checkpoint "validation-retry"

python3 ../python/plot_results.py mlem --source hawaii 2>&1 | tee "$OUT/plot_mlem.log" || true
python3 ../python/plot_results.py slices --source hawaii 2>&1 | tee "$OUT/plot_slices.log" || true
mv -f mlem_convergence_*_pool15.png slices_*_pool15.png "$OUT/" 2>/dev/null || true
checkpoint "plotting-retry"

echo ""
echo "=== retry done. all 4 configs completed, validation+plots regenerated with full data. ==="
