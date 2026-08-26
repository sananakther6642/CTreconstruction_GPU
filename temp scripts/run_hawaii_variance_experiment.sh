#!/bin/bash
# gpu-buf-speed Part B: batched, unattended test of gpu-buf's run-to-run
# timing variance mitigations, for AMD Hawaii PRO ONLY.
#
# WHY THIS SCRIPT EXISTS AND IS SHAPED THIS WAY:
# The variance is confirmed AMD-driver-specific (--diag repeat-slab is flat
# to the millisecond on the NVIDIA GTX 680, ~0.796-0.797s across 40 repeats
# -- see README.md and the session logs) and AMD Hawaii PRO (pool15-01) is
# SSH-blocked (pam_restrict_login, unresolved, confirmed multiple times
# this project). So there is no remote iteration possible -- this script
# runs everything in ONE unattended pass so a single in-person visit
# returns complete data. It has NEVER BEEN RUN. Treat every number in it
# as a plan, not a result, until it actually executes on AMD Hawaii PRO.
#
# Tests two new trigger heuristics (ct_gpu.c, both off by default) against
# the documented failed baseline mitigation:
#   FP_BUFFER_VOL_REALLOC_ON_SLOW=1   -- B1a, reacts to the GPU-BOUND signal
#                                        itself instead of guessing a schedule
#   FP_BUFFER_VOL_REALLOC_SECONDS=N   -- B1b, time-based instead of count-based
# Plus B2, a cheap diagnostic probe: does the step-degradation still appear
# on a SMALLER d_vol allocation (256^3 = 67MB vs 512^3's 537MB)? If not,
# allocation size is confirmed as the trigger and buffer-splitting becomes a
# justified follow-up; if it degrades too, size is not the lever.
#
# MEASUREMENT DISCIPLINE (the single biggest lesson from the failed
# FP_BUFFER_VOL_REALLOC_EVERY mitigation): a clean 5-epoch result completely
# failed to reproduce at 10-epoch/75-angle scale. Every arm here runs
# EPOCHS=10 at 512^3, N=3 REPEATS, and reports mean+stdev -- never trust a
# single run or a shortened epoch count for this specific problem.
#
# Run inside tmux so it survives a disconnect:
#   tmux new -s pool15-variance
#   ./run_pool15_variance_experiment.sh
#   (detach: Ctrl+B then D)
set -e

echo "=== pre-flight checks ==="

if command -v klist >/dev/null 2>&1; then
  if ! klist -s 2>/dev/null; then
    echo "WARNING: no valid Kerberos ticket. Home dir / NFS access may fail."
    echo "  Fix: kinit -R   (or kinit <username> if not renewable)"
    echo "  Continuing anyway in 5s -- Ctrl+C to abort and fix first."
    sleep 5
  else
    echo "kerberos ticket: OK"
  fi
fi

AVAIL_KB=$(df -Pk . | tail -1 | awk '{print $4}')
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
echo "disk available here: ${AVAIL_GB}GB"
if [ "$AVAIL_GB" -lt 3 ]; then
  echo "ERROR: less than 3GB free -- likely to fail partway through."
  echo "  Run ./cleanup_before_run.sh first, or free space manually."
  exit 1
fi

echo "=== pre-flight checks done ==="
echo ""

DATA512=/lgrp/edu-2026-1-gpulab/proj_512_75.hdf5
DATA256=/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5
EPOCHS=10
REPEATS=3
OUT=pool15_variance
mkdir -p "$OUT/logs"

echo "=== build (fresh, this machine's own binary) ==="
make clean && make

WORKTREE_DIR="../pool15-variance-worktree"

# Same worktree-based checkpoint pattern as run_hawaii_full.sh -- an
# in-place `git checkout` on the live working directory corrupted a real
# run earlier this project (ct_recon reading/writing the exact files git
# was switching branches on). Never touch the main working directory's
# branch from this script.
checkpoint() {
  local label="$1"
  local repo_root
  repo_root=$(git rev-parse --show-toplevel)
  (
    set +e
    if [ ! -d "$WORKTREE_DIR" ]; then
      if git show-ref --verify --quiet refs/heads/pool15-variance; then
        git worktree add "$WORKTREE_DIR" pool15-variance
      elif git ls-remote --exit-code --heads origin pool15-variance >/dev/null 2>&1; then
        git fetch origin pool15-variance
        git worktree add -B pool15-variance "$WORKTREE_DIR" origin/pool15-variance
      else
        git worktree add -B pool15-variance "$WORKTREE_DIR"
      fi
    fi
    rsync -a --delete "$repo_root/$OUT/" "$WORKTREE_DIR/$OUT/"
    cd "$WORKTREE_DIR" || exit 1
    git add -f "$OUT"/logs 2>/dev/null
    for f in "$OUT"/*.txt; do
      [ -e "$f" ] && git add -f "$f" 2>/dev/null
    done
    if git diff --cached --quiet; then
      echo "checkpoint ($label): nothing new to commit"
    else
      git commit -m "pool15 variance checkpoint: $label ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
      git push -u origin pool15-variance
    fi
  ) || echo "WARNING: checkpoint ($label) commit/push failed -- results still on disk under $OUT/, just not in git."
}

# Runs REPEATS trials of one arm, EPOCHS each, at 512^3, using --diag
# repeat-slab's underlying run_fp_buffer path via the real MLEM loop (not
# the isolated diagnostic) so results are directly comparable to the
# documented baseline [75.37, 101.89, 83.63]s / N=5-realloc
# [105.84, 86.77, 89.04, 85.31]s numbers, which were both real-MLEM-scale.
run_arm() {
  local arm_name="$1"
  shift
  local env_prefix="$*"
  echo ""
  echo "=== arm: $arm_name ($env_prefix) ==="
  local times=()
  for i in $(seq 1 "$REPEATS"); do
    local log="$OUT/logs/${arm_name}_run${i}.log"
    local t0 t1
    t0=$(date +%s)
    env $env_prefix build/ct_recon --data "$DATA512" --out /tmp/variance_test.hdf5 \
      --mode gpu-buf --epochs "$EPOCHS" --samples 512 --kernels kernels \
      2>&1 | tee "$log"
    t1=$(date +%s)
    local dt=$((t1 - t0))
    times+=("$dt")
    echo "  run $i: ${dt}s"
  done
  # mean/stdev over $REPEATS values -- printed to the arm's own summary
  # file so checkpoint() picks it up even if the script is interrupted
  # partway through a later arm.
  python3 -c "
import sys
vals = [$(IFS=,; echo "${times[*]}")]
mean = sum(vals) / len(vals)
var = sum((v - mean) ** 2 for v in vals) / len(vals)
stdev = var ** 0.5
print(f'$arm_name: n={len(vals)} mean={mean:.2f}s stdev={stdev:.2f}s vals={vals}')
" | tee -a "$OUT/summary.txt"
  rm -f /tmp/variance_test.hdf5
}

echo ""
echo "=== A: unmitigated baseline (compare against documented [75.37, 101.89, 83.63]s) ==="
run_arm "baseline" ""
checkpoint "baseline"

echo ""
echo "=== B: known-failed mitigation, re-run for a fresh same-session baseline ==="
echo "  (documented [105.84, 86.77, 89.04, 85.31]s -- expect similar, not better)"
run_arm "realloc_every5" "FP_BUFFER_VOL_REALLOC_EVERY=5"
checkpoint "realloc_every5"

echo ""
echo "=== C: B1a -- react to the GPU-BOUND signal itself (the real candidate) ==="
run_arm "realloc_on_slow" "FP_BUFFER_VOL_REALLOC_ON_SLOW=1"
checkpoint "realloc_on_slow"

echo ""
echo "=== D: B1b -- time-based trigger, 20s (roughly matches one epoch's scale) ==="
run_arm "realloc_20s" "FP_BUFFER_VOL_REALLOC_SECONDS=20"
checkpoint "realloc_20s"

echo ""
echo "=== E: B2 diagnostic probe -- does 256^3 (67MB) show the same degradation as 512^3 (537MB)? ==="
echo "  Uses --diag repeat-slab directly (isolated, not real MLEM) since"
echo "  this is a yes/no diagnostic, not a timing comparison."
build/ct_recon --data "$DATA256" --out /dev/null --mode gpu-buf \
  --diag "repeat-slab:0:8:40" --kernels kernels \
  2>&1 | tee "$OUT/logs/probe_256cubed_repeat_slab.log"
checkpoint "probe_256cubed"

echo ""
echo "=== done. Summary: ==="
cat "$OUT/summary.txt"
echo ""
echo "Full logs and summary pushed to pool15-variance branch."
echo "Compare each arm's mean against the unmitigated baseline (arm A) run"
echo "in this SAME session -- do not compare against the old documented"
echo "numbers directly, machine/driver state may have changed since."
