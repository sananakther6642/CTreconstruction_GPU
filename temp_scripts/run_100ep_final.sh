#!/bin/bash
# Full 100-epoch run, all four modes, on the current (post-optimization)
# features branch. Designed to survive an SSH drop -- launch with:
#
#   cd ~/CTreconstruction_GPU
#   nohup bash temp_scripts/run_100ep_final.sh > ct100_run.log 2>&1 &
#   disown
#
# Then check progress any time with:
#   tail -f ct100_run.log
#
# NAMING: every file this script creates or touches at the repo root is
# prefixed ct100_ (ct100_run.log, ct100_cpu_done.marker, ct100_DONE) so
# it can never collide with anything else on a shared machine, and so
# the cleanup step below can safely wipe "everything this script owns"
# with one glob instead of an easy-to-miss explicit filename list.
# Actual reconstruction output stays inside the timestamped $OUTDIR,
# per-run, never overwritten.
#
# run_topic2_20ep.sh polls for ct100_cpu_done.marker, written right
# after the CPU pass (before gpu-buf/img/opt start) -- it only needs
# CPU done, not the whole script, per the "wait for CPU then run
# Topic_2" ask.
set -e
cd "$(dirname "$0")/.."

DATA=/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5
EPOCHS=100
OUTDIR=results_100ep_$(date +%Y%m%d_%H%M%S)

echo "=== $(date) : cleanup before run ==="
# Everything this script (and run_topic2_20ep.sh) owns, by prefix --
# stale markers/sentinels from an earlier run would make a re-run, or
# run_topic2_20ep.sh, think a step already finished when it didn't.
rm -f ct100_*.marker ct100_*.DONE ct100_run.log ct_topic2_*.log ct_topic2_*.DONE
# Stale root-level output_*.hdf5 from earlier ad-hoc make run-cpu/
# run-gpu-* testing this session -- not this script's own outputs
# (those go in $OUTDIR, timestamped, never overwritten), but leaving
# old ones at the root risks someone reading the wrong file.
rm -f output_cpu.hdf5 output_gpu_buf.hdf5 output_gpu_img.hdf5 \
      output_gpu_opt.hdf5 output_gpu_opt_512.hdf5 output_cpu_512.hdf5 \
      output_gpu_buf_512.hdf5 output_gpu_img_512.hdf5
# Build artifacts -- 'make clean && make' below rebuilds these, but
# clear first so a partial/interrupted previous build can't linger.
rm -rf build
echo "cleanup done."

mkdir -p "$OUTDIR"

echo "=== $(date) : build ==="
make clean && make

echo ""
echo "=== $(date) : CPU mode, ${EPOCHS} epochs ==="
OMP_NUM_THREADS=$(nproc) OMP_PROC_BIND=close OMP_PLACES=cores \
  build/ct_recon --data "$DATA" --out "$OUTDIR/output_cpu.hdf5" \
    --mode cpu --epochs "$EPOCHS" --kernels kernels \
    2>&1 | tee "$OUTDIR/cpu.log"
echo "$(date)" > ct100_cpu_done.marker

echo ""
echo "=== $(date) : gpu-buf mode, ${EPOCHS} epochs ==="
build/ct_recon --data "$DATA" --out "$OUTDIR/output_gpu_buf.hdf5" \
  --mode gpu-buf --epochs "$EPOCHS" --kernels kernels \
  2>&1 | tee "$OUTDIR/gpu_buf.log"

echo ""
echo "=== $(date) : gpu-img mode, ${EPOCHS} epochs ==="
build/ct_recon --data "$DATA" --out "$OUTDIR/output_gpu_img.hdf5" \
  --mode gpu-img --epochs "$EPOCHS" --kernels kernels \
  2>&1 | tee "$OUTDIR/gpu_img.log"

echo ""
echo "=== $(date) : gpu-opt mode (S=1), ${EPOCHS} epochs ==="
build/ct_recon --data "$DATA" --out "$OUTDIR/output_gpu_opt.hdf5" \
  --mode gpu-opt --epochs "$EPOCHS" --kernels kernels \
  2>&1 | tee "$OUTDIR/gpu_opt_s1.log"

echo ""
echo "=== $(date) : gpu-opt mode (S=5, OSEM), ${EPOCHS} epochs ==="
build/ct_recon --data "$DATA" --out "$OUTDIR/output_gpu_opt_s5.hdf5" \
  --mode gpu-opt --subsets 5 --epochs "$EPOCHS" --kernels kernels \
  2>&1 | tee "$OUTDIR/gpu_opt_s5.log"

echo ""
echo "=== $(date) : validate (MSE vs CPU) ==="
VALIDATE_DIR="$OUTDIR" python3 python/validate.py 2>&1 | tee "$OUTDIR/validate.log" || \
  echo "validate.py may expect different filenames -- check $OUTDIR manually if this failed"

echo ""
echo "=== $(date) : all done. Results in $OUTDIR/ ==="
echo "$(date)" > ct100_DONE
echo "$OUTDIR" >> ct100_DONE
