#!/bin/bash
# Full 100-epoch run, all four modes, on the current (post-optimization)
# features branch. Designed to survive an SSH drop -- launch with:
#
#   cd ~/CTreconstruction_GPU
#   nohup bash temp_scripts/run_100ep_final.sh > run_100ep_final.log 2>&1 &
#   disown
#
# Then check progress any time with:
#   tail -f run_100ep_final.log
#
# A sentinel file (run_100ep_final.DONE) is written on successful
# completion -- run_topic2_20ep.sh polls for it. cpu_done.marker is
# written right after the cpu run specifically, so that script can start
# even before gpu-buf/img/opt finish (it only needs CPU to be done, per
# the "wait for CPU then run Topic_2" ask -- Topic_2 is CPU-bound Python
# and would otherwise contend with the GPU runs' host-side work anyway,
# but the two are independent, so no need to wait for the whole script).
set -e
cd "$(dirname "$0")/.."

DATA=/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5
EPOCHS=100
OUTDIR=results_100ep_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTDIR"

echo "=== $(date) : build ==="
make clean && make

echo ""
echo "=== $(date) : CPU mode, ${EPOCHS} epochs ==="
OMP_NUM_THREADS=$(nproc) OMP_PROC_BIND=close OMP_PLACES=cores \
  build/ct_recon --data "$DATA" --out "$OUTDIR/output_cpu.hdf5" \
    --mode cpu --epochs "$EPOCHS" --kernels kernels \
    2>&1 | tee "$OUTDIR/cpu.log"
touch cpu_done.marker
echo "$(date)" > cpu_done.marker

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
echo "$(date)" > run_100ep_final.DONE
echo "$OUTDIR" >> run_100ep_final.DONE
