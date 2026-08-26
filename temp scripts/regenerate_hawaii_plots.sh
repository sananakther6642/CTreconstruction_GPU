#!/bin/bash
# ARCHIVED: paths assume this script runs from repo root (its original
# location before the temp scripts/ cleanup move) -- "../python/X.py" and
# any "../*-worktree" paths need re-checking before rerunning from here.
# Standalone: regenerate validate.py + plot_results.py output for AMD Hawaii PRO
# once all 8 C/GPU configs (4 modes x 2 sizes) are present on disk, then
# checkpoint the result to pool15-results the same way run_hawaii_full.sh
# and retry_hawaii_gpu_configs.sh do. Safe to run any time after confirming
# all 8 pool15/hdf5/*.hdf5 files exist -- checks first and refuses to run
# on incomplete data instead of silently producing partial plots again.
#
# Run inside tmux if it might outlive your session:
#   tmux new -s pool15-plots
#   ./regenerate_hawaii_plots.sh
set -e

OUT=pool15
REQUIRED="cpu_256 gpu-buf_256 gpu-img_256 gpu-opt_256 cpu_512 gpu-buf_512 gpu-img_512 gpu-opt_512"

echo "=== checking all 8 configs are present ==="
MISSING=0
for tag in $REQUIRED; do
  if [ ! -f "$OUT/hdf5/${tag}.hdf5" ]; then
    echo "MISSING: $OUT/hdf5/${tag}.hdf5"
    MISSING=1
  fi
done
if [ "$MISSING" -eq 1 ]; then
  echo "Refusing to regenerate plots with missing configs -- run"
  echo "retry_hawaii_gpu_configs.sh (or the missing run_one calls) first."
  exit 1
fi
echo "all 8 configs present."

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
    for f in "$OUT"/*.txt "$OUT"/*.png; do
      [ -e "$f" ] && git add -f "$f" 2>/dev/null
    done
    if git diff --cached --quiet; then
      echo "checkpoint ($label): nothing new to commit"
    else
      git commit -m "pool15 checkpoint: $label ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
      git push -u origin pool15-results
    fi
  ) || echo "WARNING: checkpoint ($label) commit/push failed -- results still on disk under $OUT/, just not in git."
}

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
python3 ../python/validate.py     2>&1 | tee "$OUT/validate_256.txt" || true
python3 ../python/validate.py 512 2>&1 | tee "$OUT/validate_512.txt" || true
checkpoint "validation-full"

echo ""
echo "=== plotting figures (--source hawaii) ==="
python3 ../python/plot_results.py mlem --source hawaii 2>&1 | tee "$OUT/plot_mlem.log" || true
python3 ../python/plot_results.py slices --source hawaii 2>&1 | tee "$OUT/plot_slices.log" || true
mv -f mlem_convergence_*_pool15.png slices_*_pool15.png "$OUT/" 2>/dev/null || true
checkpoint "plotting-full"

echo ""
echo "=== done. validation + plots regenerated with all 8 configs, pushed to pool15-results. ==="
