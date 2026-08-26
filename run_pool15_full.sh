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

echo "=== pre-flight checks ==="

# Kerberos ticket -- home dir/NFS access dies silently without one
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

# Disk space -- need room for ~8 output volumes (256^3 ~65MB each,
# 512^3 ~513MB each) plus logs/csv, call it ~2.5GB with margin
AVAIL_KB=$(df -Pk . | tail -1 | awk '{print $4}')
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
echo "disk available here: ${AVAIL_GB}GB"
if [ "$AVAIL_GB" -lt 3 ]; then
  echo "ERROR: less than 3GB free -- likely to fail partway through."
  echo "  Run ./cleanup_before_run.sh first, or free space manually."
  exit 1
fi

# Memory -- cpu-512 alone needs a few GB resident; warn if the machine
# is under heavy pressure (this is what killed cpu-512 on kale)
if command -v free >/dev/null 2>&1; then
  AVAIL_MEM_GB=$(free -g | awk '/^Mem:/{print $7}')
  echo "memory available: ${AVAIL_MEM_GB}GB"
  if [ -n "$AVAIL_MEM_GB" ] && [ "$AVAIL_MEM_GB" -lt 4 ]; then
    echo "WARNING: less than 4GB free RAM. cpu-512 in particular may get"
    echo "  OOM-killed partway through (this happened on kale). Check"
    echo "  'ps aux --sort=-%mem | head' for what's using memory before"
    echo "  committing to an unattended run, or proceed and just rerun"
    echo "  this script later -- it's resumable and will pick up where"
    echo "  it left off."
    echo "  Continuing anyway in 5s -- Ctrl+C to abort and check first."
    sleep 5
  fi
fi

echo "=== pre-flight checks done ==="
echo ""

DATA256=/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5
DATA512=/lgrp/edu-2026-1-gpulab/proj_512_75.hdf5
EPOCHS=100
OUT=pool15
mkdir -p "$OUT/hdf5" "$OUT/conv_csv" "$OUT/logs"

is_complete() {
  local csv="$1"
  [ -f "$csv" ] && [ "$(tail -1 "$csv" | cut -d, -f1)" = "$EPOCHS" ]
}

# Commit+push whatever's landed under $OUT so far to a dedicated results
# branch. Called after every stage below -- pool15-01 isn't remotely
# reachable, so partial progress must reach git incrementally rather than
# in one commit at the very end (a mid-run reboot/hang would otherwise
# lose everything already finished). .hdf5 volumes are never committed
# (2.4GB+, already globally gitignored) -- only logs/CSVs/PNGs/txt.
checkpoint() {
  local label="$1"
  (
    set +e
    STARTING_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    # -B would reset pool15-results to the current branch's HEAD every call,
    # discarding every earlier checkpoint's commit. Create it once (from
    # whatever local/remote copy already exists, or fresh off HEAD if this
    # is truly the first run), then just switch onto it on later calls so
    # commits accumulate.
    if git show-ref --verify --quiet refs/heads/pool15-results; then
      git checkout pool15-results
    elif git ls-remote --exit-code --heads origin pool15-results >/dev/null 2>&1; then
      git fetch origin pool15-results
      git checkout -B pool15-results origin/pool15-results
    else
      git checkout -B pool15-results
    fi
    # One git add per path -- a single command with multiple globs means
    # any glob that matches nothing (e.g. no *.txt yet on the first
    # checkpoint) makes bash pass the literal unexpanded string, which git
    # add rejects with "pathspec did not match any files" and a nonzero
    # exit -- killing the ENTIRE add, including paths that did match.
    # Verified: this silently dropped cpu_256's csv/log on pool15-01's
    # first real checkpoint ("nothing new to commit" despite real output
    # existing on disk).
    git add -f "$OUT"/logs 2>/dev/null
    git add -f "$OUT"/conv_csv 2>/dev/null
    git add -f "$OUT"/hardware.txt 2>/dev/null
    git add -f "$OUT"/topic2_python.log 2>/dev/null
    for f in "$OUT"/*.txt "$OUT"/*.png; do
      [ -e "$f" ] && git add -f "$f" 2>/dev/null
    done
    if git diff --cached --quiet; then
      echo "checkpoint ($label): nothing new to commit"
    else
      git commit -m "pool15 checkpoint: $label ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
      git push -u origin pool15-results
    fi
    git checkout "$STARTING_BRANCH"
  ) || echo "WARNING: checkpoint ($label) commit/push failed -- results still on disk under $OUT/, just not in git."
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
  checkpoint "${MODE}_256"
done

echo ""
echo "=== 512^3, GPU modes (fast, run before cpu) ==="
for MODE in gpu-buf gpu-img gpu-opt; do
  run_one "$MODE" 512 "$DATA512" "--samples 512"
  checkpoint "${MODE}_512"
done

echo ""
echo "=== 512^3, cpu (slow, ~1hr) ==="
run_one cpu 512 "$DATA512" "--samples 512"
checkpoint "cpu_512"

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
checkpoint "validation"

echo ""
echo "=== plotting figures (--source pool15) ==="
python3 plot_results.py mlem --source pool15 2>&1 | tee "$OUT/plot_mlem.log" || true
python3 plot_results.py slices --source pool15 2>&1 | tee "$OUT/plot_slices.log" || true
mv -f mlem_convergence_*_pool15.png slices_*_pool15.png "$OUT/" 2>/dev/null || true
checkpoint "plotting"

echo ""
echo "=== Python reference (Topic_2_CTreconstruction.py), 256^3, 100 epochs ==="
echo "  hardcoded to 256^3, pure Python fp/bp -- slow (~3679s/epoch measured"
echo "  on kale at sample_ratio=1, vs ~4690s/epoch at sample_ratio=2), runs"
echo "  last so it never blocks the faster C/GPU configs above."
echo "  100 epochs ~= 4.3 days -- runs unattended, check back later."
python3 Topic_2_CTreconstruction.py 2>&1 | tee "$OUT/topic2_python.log" || true
if [ -f output_python_reconstruction.hdf5 ]; then
  mv -f output_python_reconstruction.hdf5 "$OUT/hdf5/python_256.hdf5"
  echo "Saved: $OUT/hdf5/python_256.hdf5"
fi
checkpoint "topic2"

echo ""
echo "=== done. Everything under $OUT/ ==="
find "$OUT" -type f | sort
