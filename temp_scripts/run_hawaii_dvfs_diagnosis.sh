#!/bin/bash
# gpu-buf DVFS diagnosis on AMD Hawaii PRO (pool15-01).
#
# Re-tests the README's "gpu-buf run-to-run variance" root cause. The
# documented explanation (AMD-driver memory-placement demotion of the
# 537MB d_vol buffer, "recovers on realloc, degrades again after 6-13
# launches") is contradicted by a 30-epoch log showing the slowdown
# self-recovers with NO reallocation (epoch 8: 64.78s -> epoch 9: 15.77s,
# fully back to baseline). See src/ct_gpu.c's run_fp_buffer comment and
# README.md's gpu-buf variance section for the full history.
#
# Primary hypothesis under test here: memory-clock DVFS. fp_buffer is
# bandwidth-bound (uncoalesced trilinear_buf); Hawaii's mclk has an 8.3x
# DPM range (1250->150 MHz) vs the ~3.2x core-clock range the earlier
# "not thermal" argument used -- comfortably covers the observed ~6.6x
# slowdown, and predicts exactly the mode-selectivity already observed
# (gpu-img/gpu-opt use the texture cache, far less mclk-sensitive).
#
# Requires: this build with the FP_BUFFER_CLOCK_PROBE sysfs probe
# (sample_gpu_clocks in src/ct_gpu.c) and the repeat-slab lws fix (both
# added alongside this script -- rebuild before running).
#
# Run inside tmux so it survives a disconnect:
#   tmux new -s dvfs
#   ./run_hawaii_dvfs_diagnosis.sh
#   (detach: Ctrl+B then D; reattach: tmux attach -t dvfs)
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

# sysfs probe readability check -- if these aren't readable, arm A's clock
# column will be blank ("sclk=? mclk=? temp=?") and the DVFS hypothesis
# can't be confirmed/denied from this run; still worth running for the
# self-recovery timing alone, but flag it up front.
CARD_IDX="${FP_BUFFER_CLOCK_PROBE_CARD:-0}"
SCLK_PATH="/sys/class/drm/card${CARD_IDX}/device/pp_dpm_sclk"
MCLK_PATH="/sys/class/drm/card${CARD_IDX}/device/pp_dpm_mclk"
if [ -r "$SCLK_PATH" ] && [ -r "$MCLK_PATH" ]; then
  echo "sysfs DPM tables readable: OK ($SCLK_PATH, $MCLK_PATH)"
else
  echo "WARNING: cannot read $SCLK_PATH / $MCLK_PATH."
  echo "  Try FP_BUFFER_CLOCK_PROBE_CARD=<N> for a different card index,"
  echo "  or check 'ls /sys/class/drm/'. Continuing anyway in 5s."
  sleep 5
fi

echo "=== pre-flight checks done ==="
echo ""

DATA512=/lgrp/edu-2026-1-gpulab/proj_512_75.hdf5
EPOCHS=30
OUT=dvfs_diag
mkdir -p "$OUT/logs" "$OUT/samplers"

# ── Independent sysfs sampler: polls clocks/temp + who's on the GPU at a
# finer grain (200ms) than the in-kernel probe (which only fires on slabs
# already measured >2s), so the fast->slow transition itself is caught,
# not just steady-state readings once a burst is already underway. Also
# covers arm B (contention control): logs which PIDs hold /dev/dri fds.
sampler_pid=""
start_sampler() {
  local out_csv="$1"
  {
    echo "t_unix,sclk,mclk,temp_c,dri_pids"
    while true; do
      local sclk mclk temp pids
      sclk=$(grep '\*' "$SCLK_PATH" 2>/dev/null | awk '{print $2}')
      mclk=$(grep '\*' "$MCLK_PATH" 2>/dev/null | awk '{print $2}')
      temp=$(cat /sys/class/drm/card${CARD_IDX}/device/hwmon/hwmon*/temp1_input 2>/dev/null | head -1)
      pids=$(fuser /dev/dri/card${CARD_IDX} 2>/dev/null | tr -s ' ' ';')
      echo "$(date +%s.%N),${sclk:-?},${mclk:-?},${temp:-?},\"${pids:-}\""
      sleep 0.2
    done
  } > "$out_csv" &
  sampler_pid=$!
}
stop_sampler() {
  if [ -n "$sampler_pid" ]; then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
    sampler_pid=""
  fi
}
trap stop_sampler EXIT

WORKTREE_DIR="../pool15-dvfs-worktree"
checkpoint() {
  local label="$1"
  local repo_root
  repo_root=$(git rev-parse --show-toplevel)
  (
    set +e
    if [ ! -d "$WORKTREE_DIR" ]; then
      if git show-ref --verify --quiet refs/heads/pool15-dvfs-diagnosis; then
        git worktree add "$WORKTREE_DIR" pool15-dvfs-diagnosis
      elif git ls-remote --exit-code --heads origin pool15-dvfs-diagnosis >/dev/null 2>&1; then
        git fetch origin pool15-dvfs-diagnosis
        git worktree add -B pool15-dvfs-diagnosis "$WORKTREE_DIR" origin/pool15-dvfs-diagnosis
      else
        git worktree add -B pool15-dvfs-diagnosis "$WORKTREE_DIR"
      fi
    fi
    rsync -a --delete "$repo_root/$OUT/" "$WORKTREE_DIR/$OUT/"
    cd "$WORKTREE_DIR" || exit 1
    git add -f "$OUT" 2>/dev/null
    if git diff --cached --quiet; then
      echo "checkpoint ($label): nothing new to commit"
    else
      git commit -m "dvfs-diag checkpoint: $label ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
      git push -u origin pool15-dvfs-diagnosis
    fi
  ) || echo "WARNING: checkpoint ($label) commit/push failed -- results still on disk under $OUT/, just not in git."
}

echo "=== hardware ===" | tee "$OUT/hardware.txt"
hostname | tee -a "$OUT/hardware.txt"
clinfo 2>/dev/null | grep -i "device name" | head -1 | tee -a "$OUT/hardware.txt"
cat "$SCLK_PATH" 2>/dev/null | tee -a "$OUT/hardware.txt"
cat "$MCLK_PATH" 2>/dev/null | tee -a "$OUT/hardware.txt"

echo ""
echo "=== Arm A: instrumented gpu-buf baseline, 512^3, $EPOCHS epochs ==="
echo "    (H1 verdict: do slow slabs show a low mclk DPM state?)"
start_sampler "$OUT/samplers/armA_sampler.csv"
FP_BUFFER_CLOCK_PROBE=1 build/ct_recon --data "$DATA512" \
  --out "$OUT/armA_gpu-buf_512.hdf5" --mode gpu-buf --epochs $EPOCHS \
  --samples 512 --kernels kernels 2>&1 | tee "$OUT/logs/armA_gpu-buf_512.log"
stop_sampler
checkpoint "armA"

echo ""
echo "=== Arm B: contention control (who else is on the GPU during bursts) ==="
echo "    -- covered by the dri_pids column in armA_sampler.csv already;"
echo "    this arm is just a targeted re-read of that column for anyone"
echo "    other than this run's own PID. No separate run needed."
grep -v "^t_unix" "$OUT/samplers/armA_sampler.csv" | awk -F',' '{print $5}' | sort -u > "$OUT/armB_dri_pids_seen.txt"
echo "distinct dri_pids field values seen during Arm A:" | tee -a "$OUT/armB_dri_pids_seen.txt"
cat "$OUT/armB_dri_pids_seen.txt"
checkpoint "armB"

echo ""
echo "=== Arm C: mode-selectivity control, gpu-img, 512^3, $EPOCHS epochs ==="
echo "    (H1 predicts this stays flat even if the sampler shows mclk dropping --"
echo "    that separates 'GPU downclocked' from 'GPU busy with someone else's work',"
echo "    since contention (H2) would slow both modes together.)"
start_sampler "$OUT/samplers/armC_sampler.csv"
build/ct_recon --data "$DATA512" \
  --out "$OUT/armC_gpu-img_512.hdf5" --mode gpu-img --epochs $EPOCHS \
  --samples 512 --kernels kernels 2>&1 | tee "$OUT/logs/armC_gpu-img_512.log"
stop_sampler
checkpoint "armC"

echo ""
echo "=== Arm D: fixed repeat-slab (post-lws-fix), clock probe on ==="
echo "    (re-establishes step-and-hold vs self-recovering burst at the"
echo "    PRODUCTION lws={2,16,2}, not the old {4,64,1} mismatch)"
FP_BUFFER_CLOCK_PROBE=1 build/ct_recon --data "$DATA512" \
  --mode gpu-buf --kernels kernels \
  --diag repeat-slab:0:8:60 2>&1 | tee "$OUT/logs/armD_repeat-slab.log"
checkpoint "armD"

echo ""
echo "=== done. Everything under $OUT/ ==="
find "$OUT" -type f | sort
echo ""
echo "Verdict check:"
echo "  1. grep for 'SLOW' lines in armA log -- do they cluster around a"
echo "     low mclk reading vs the baseline probe's reading?"
echo "  2. Compare armA_sampler.csv mclk column against the log's slab"
echo "     timestamps (t= field, once ct_gpu.c prints one -- see plan)."
echo "  3. Did armC (gpu-img) stay flat across the whole run regardless of"
echo "     what armA/armC's own sampler shows for mclk during that window?"
echo "  4. armD: does the step degradation now RECOVER ON ITS OWN before"
echo "     realloc_at, at the corrected lws? (matches the 30-epoch log's"
echo "     self-recovery, or still holds until realloc as previously reported?)"
