#!/bin/bash
# Full 100-epoch run at 512^3, all modes, on the current (post-optimization)
# features branch. Meant to run overnight inside tmux -- see the tmux
# instructions in the commit/PR message or ask Claude for the exact
# session commands.
#
# Order matters here: cpu (~57 min) and gpu-buf/img/opt-S1 (~15 min
# combined) are the results that matter most and run first, each saved
# to disk before the next starts. OSEM S=2 (confirmed feasible this
# session, VRAM fits) runs next. S=3 runs LAST, deliberately, and is
# allowed to fail -- 512^3's VRAM budget analysis calculated S=3 as
# exceeding the GTX 680's 4037 MiB limit and it has never been tested;
# if it crashes, hangs, or resets the driver, everything before it in
# this script has already been safely written to disk and is
# unaffected. Check S=3's own log/exit status before trusting its
# output at all.
#
# NAMING: prefixed ct512_ throughout, same rationale as
# run_100ep_final.sh's ct100_ prefix -- collision-safe on a shared
# machine, one glob to clean up. ct512_cpu_done.marker is written after
# the CPU pass specifically (before GPU modes start), matching the
# ct100_ pattern, in case something downstream wants to key off it.
set -e
cd "$(dirname "$0")/.."

DATA512=/lgrp/edu-2026-1-gpulab/proj_512_75.hdf5
SAMPLES512=512
EPOCHS=100
OUTDIR=results_512_100ep_$(date +%Y%m%d_%H%M%S)

echo "=== $(date) : cleanup before run ==="
rm -f ct512_*.marker ct512_*.DONE ct512_run.log
rm -f output_cpu_512.hdf5 output_gpu_buf_512.hdf5 output_gpu_img_512.hdf5 output_gpu_opt_512.hdf5
rm -rf build
echo "cleanup done."

mkdir -p "$OUTDIR"

echo "=== $(date) : build ==="
make clean && make
ls -la build/ct_recon   # fail loudly and visibly if the binary isn't actually there

echo ""
echo "=== $(date) : CPU mode, 512^3, ${EPOCHS} epochs (expect ~55-60 min) ==="
OMP_NUM_THREADS=$(nproc) OMP_PROC_BIND=close OMP_PLACES=cores \
  build/ct_recon --data "$DATA512" --out "$OUTDIR/output_cpu_512.hdf5" \
    --mode cpu --epochs "$EPOCHS" --samples "$SAMPLES512" --kernels kernels \
    2>&1 | tee "$OUTDIR/cpu.log"
echo "$(date)" > ct512_cpu_done.marker

echo ""
echo "=== $(date) : gpu-buf mode, 512^3, ${EPOCHS} epochs ==="
build/ct_recon --data "$DATA512" --out "$OUTDIR/output_gpu_buf_512.hdf5" \
  --mode gpu-buf --epochs "$EPOCHS" --samples "$SAMPLES512" --kernels kernels \
  2>&1 | tee "$OUTDIR/gpu_buf.log"

echo ""
echo "=== $(date) : gpu-img mode, 512^3, ${EPOCHS} epochs ==="
build/ct_recon --data "$DATA512" --out "$OUTDIR/output_gpu_img_512.hdf5" \
  --mode gpu-img --epochs "$EPOCHS" --samples "$SAMPLES512" --kernels kernels \
  2>&1 | tee "$OUTDIR/gpu_img.log"

echo ""
echo "=== $(date) : gpu-opt mode (S=1), 512^3, ${EPOCHS} epochs ==="
build/ct_recon --data "$DATA512" --out "$OUTDIR/output_gpu_opt_512.hdf5" \
  --mode gpu-opt --epochs "$EPOCHS" --samples "$SAMPLES512" --kernels kernels \
  2>&1 | tee "$OUTDIR/gpu_opt_s1.log"

echo ""
echo "=== $(date) : validate (256^3-shaped modes vs CPU; 512^3 needs --dir) ==="
VALIDATE_DIR="$OUTDIR" python3 python/validate.py 512 2>&1 | tee "$OUTDIR/validate.log" || \
  echo "validate.py 512 may need different filenames/args -- check $OUTDIR manually if this failed"

echo ""
echo "=== $(date) : gpu-opt OSEM S=2, 512^3, ${EPOCHS} epochs (confirmed feasible this session) ==="
build/ct_recon --data "$DATA512" --out "$OUTDIR/output_gpu_opt_512_s2.hdf5" \
  --mode gpu-opt --subsets 2 --epochs "$EPOCHS" --samples "$SAMPLES512" --kernels kernels \
  2>&1 | tee "$OUTDIR/gpu_opt_s2.log"

echo "$(date)" > ct512_DONE
echo "$OUTDIR" >> ct512_DONE
echo "=== $(date) : everything through S=2 is done and safely on disk. ==="

echo ""
echo "=== $(date) : gpu-opt OSEM S=3, 512^3, ${EPOCHS} epochs -- UNTESTED, may fail/hang/crash. ==="
echo "=== Calculated to exceed the GTX 680's 4037 MiB VRAM budget; running anyway to see the actual failure mode. ==="
echo "=== Everything above this line is already complete and unaffected by whatever happens next. ==="
set +e   # don't let an S=3 failure take down the rest of the script's own exit status
build/ct_recon --data "$DATA512" --out "$OUTDIR/output_gpu_opt_512_s3.hdf5" \
  --mode gpu-opt --subsets 3 --epochs "$EPOCHS" --samples "$SAMPLES512" --kernels kernels \
  2>&1 | tee "$OUTDIR/gpu_opt_s3_ATTEMPT.log"
S3_EXIT=$?
set -e

echo ""
if [ "$S3_EXIT" -eq 0 ]; then
    echo "=== $(date) : S=3 completed with exit 0 -- check gpu_opt_s3_ATTEMPT.log and the output file before trusting it. ==="
else
    echo "=== $(date) : S=3 failed (exit $S3_EXIT), as anticipated. See gpu_opt_s3_ATTEMPT.log. Everything else in this run is unaffected. ==="
fi

echo "$(date)" > ct512_ALL_DONE
echo "$OUTDIR" >> ct512_ALL_DONE
echo "S=3 exit code: $S3_EXIT" >> ct512_ALL_DONE
echo "=== $(date) : full script complete (including the S=3 attempt). Results in $OUTDIR/ ==="
