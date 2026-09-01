#!/bin/bash
# Fills every gap remaining on kale after this session's optimization work:
#   1. CLI, 512^3, all modes, 100 epochs (cpu/gpu-buf/gpu-img/gpu-opt-S1/
#      gpu-opt-S2 -- S2, not S5, since S>=3 was calculated infeasible at
#      512^3 and S=2 is the confirmed-working OSEM count there)
#   2. pybind, 256^3, all modes, 100 epochs (pybind is confirmed hump-free
#      in CPU mode over a real 40-epoch run this session -- this is the
#      full 100-epoch reference set for the report)
#   3. pybind, 512^3, all modes, 100 epochs
#
# CLI's 256^3 100-epoch set already exists (results_100ep_20260901_022731/,
# already copied to submission_outputs/gtx680/) and is NOT re-run here.
#
# NAMING: ct_kalefull_ prefix throughout, same collision-safety rationale
# as ct100_/ct512_/ct_topic2_ in the other scripts this session.
#
# Launch with (foreground is fine -- total runtime estimate below; use
# tmux/nohup if you want to detach):
#   cd ~/CTreconstruction_GPU
#   bash temp_scripts/run_kale_full_100ep.sh 2>&1 | tee ct_kalefull_run.log
set -e
cd "$(dirname "$0")/.."

DATA256=/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5
DATA512=/lgrp/edu-2026-1-gpulab/proj_512_75.hdf5
EPOCHS=100
OUTDIR=results_kalefull_100ep_$(date +%Y%m%d_%H%M%S)

echo "=== $(date) : cleanup before run ==="
rm -f ct_kalefull_*.marker ct_kalefull_*.DONE
rm -f output_cpu_512.hdf5 output_gpu_buf_512.hdf5 output_gpu_img_512.hdf5 output_gpu_opt_512.hdf5
rm -rf ~/.cache/torch_extensions/ct_recon-*
rm -rf build
echo "cleanup done."

mkdir -p "$OUTDIR"

echo "=== $(date) : build (CLI) ==="
make clean && make
ls -la build/ct_recon

# ── Part 1: CLI, 512^3, all modes ──────────────────────────────────────
echo ""
echo "=== $(date) : [1/3] CLI 512^3 cpu, ${EPOCHS} epochs (expect ~55-60 min) ==="
OMP_NUM_THREADS=$(nproc) OMP_PROC_BIND=close OMP_PLACES=cores \
  build/ct_recon --data "$DATA512" --out "$OUTDIR/cli_output_cpu_512.hdf5" \
    --mode cpu --epochs "$EPOCHS" --samples 512 --kernels kernels \
    2>&1 | tee "$OUTDIR/cli_cpu_512.log"

echo ""
echo "=== $(date) : [1/3] CLI 512^3 gpu-buf, ${EPOCHS} epochs ==="
build/ct_recon --data "$DATA512" --out "$OUTDIR/cli_output_gpu_buf_512.hdf5" \
  --mode gpu-buf --epochs "$EPOCHS" --samples 512 --kernels kernels \
  2>&1 | tee "$OUTDIR/cli_gpu_buf_512.log"

echo ""
echo "=== $(date) : [1/3] CLI 512^3 gpu-img, ${EPOCHS} epochs ==="
build/ct_recon --data "$DATA512" --out "$OUTDIR/cli_output_gpu_img_512.hdf5" \
  --mode gpu-img --epochs "$EPOCHS" --samples 512 --kernels kernels \
  2>&1 | tee "$OUTDIR/cli_gpu_img_512.log"

echo ""
echo "=== $(date) : [1/3] CLI 512^3 gpu-opt (S=1), ${EPOCHS} epochs ==="
build/ct_recon --data "$DATA512" --out "$OUTDIR/cli_output_gpu_opt_512.hdf5" \
  --mode gpu-opt --epochs "$EPOCHS" --samples 512 --kernels kernels \
  2>&1 | tee "$OUTDIR/cli_gpu_opt_s1_512.log"

echo ""
echo "=== $(date) : [1/3] CLI 512^3 gpu-opt OSEM S=2, ${EPOCHS} epochs (confirmed feasible this session; S>=3 skipped, calculated infeasible) ==="
build/ct_recon --data "$DATA512" --out "$OUTDIR/cli_output_gpu_opt_512_s2.hdf5" \
  --mode gpu-opt --subsets 2 --epochs "$EPOCHS" --samples 512 --kernels kernels \
  2>&1 | tee "$OUTDIR/cli_gpu_opt_s2_512.log"

echo "$(date)" > ct_kalefull_cli512_done.marker

# ── Part 2: pybind, 256^3, all modes ───────────────────────────────────
echo ""
echo "=== $(date) : [2/3] pybind 256^3 cpu, ${EPOCHS} epochs (hump-free, confirmed this session) ==="
python3 -m pybind_backend.run --data "$DATA256" --out "$OUTDIR/pybind_output_cpu_256.hdf5" \
  --mode cpu --epochs "$EPOCHS" \
  2>&1 | tee "$OUTDIR/pybind_cpu_256.log"

echo ""
echo "=== $(date) : [2/3] pybind 256^3 gpu-buf, ${EPOCHS} epochs ==="
python3 -m pybind_backend.run --data "$DATA256" --out "$OUTDIR/pybind_output_gpu_buf_256.hdf5" \
  --mode gpu-buf --epochs "$EPOCHS" \
  2>&1 | tee "$OUTDIR/pybind_gpu_buf_256.log"

echo ""
echo "=== $(date) : [2/3] pybind 256^3 gpu-img, ${EPOCHS} epochs ==="
python3 -m pybind_backend.run --data "$DATA256" --out "$OUTDIR/pybind_output_gpu_img_256.hdf5" \
  --mode gpu-img --epochs "$EPOCHS" \
  2>&1 | tee "$OUTDIR/pybind_gpu_img_256.log"

echo ""
echo "=== $(date) : [2/3] pybind 256^3 gpu-opt (S=1), ${EPOCHS} epochs ==="
python3 -m pybind_backend.run --data "$DATA256" --out "$OUTDIR/pybind_output_gpu_opt_256.hdf5" \
  --mode gpu-opt --epochs "$EPOCHS" \
  2>&1 | tee "$OUTDIR/pybind_gpu_opt_s1_256.log"

echo ""
echo "=== $(date) : [2/3] pybind 256^3 gpu-opt OSEM S=5, ${EPOCHS} epochs ==="
python3 -m pybind_backend.run --data "$DATA256" --out "$OUTDIR/pybind_output_gpu_opt_256_s5.hdf5" \
  --mode gpu-opt --subsets 5 --epochs "$EPOCHS" \
  2>&1 | tee "$OUTDIR/pybind_gpu_opt_s5_256.log"

echo "$(date)" > ct_kalefull_pybind256_done.marker

# ── Part 3: pybind, 512^3, all modes ───────────────────────────────────
echo ""
echo "=== $(date) : [3/3] pybind 512^3 cpu, ${EPOCHS} epochs (expect ~55-60 min; verified flat/no-hump over 40ep this session) ==="
python3 -m pybind_backend.run --data "$DATA512" --out "$OUTDIR/pybind_output_cpu_512.hdf5" \
  --mode cpu --epochs "$EPOCHS" --samples 512 \
  2>&1 | tee "$OUTDIR/pybind_cpu_512.log"

echo ""
echo "=== $(date) : [3/3] pybind 512^3 gpu-buf, ${EPOCHS} epochs ==="
python3 -m pybind_backend.run --data "$DATA512" --out "$OUTDIR/pybind_output_gpu_buf_512.hdf5" \
  --mode gpu-buf --epochs "$EPOCHS" --samples 512 \
  2>&1 | tee "$OUTDIR/pybind_gpu_buf_512.log"

echo ""
echo "=== $(date) : [3/3] pybind 512^3 gpu-img, ${EPOCHS} epochs ==="
python3 -m pybind_backend.run --data "$DATA512" --out "$OUTDIR/pybind_output_gpu_img_512.hdf5" \
  --mode gpu-img --epochs "$EPOCHS" --samples 512 \
  2>&1 | tee "$OUTDIR/pybind_gpu_img_512.log"

echo ""
echo "=== $(date) : [3/3] pybind 512^3 gpu-opt (S=1), ${EPOCHS} epochs ==="
python3 -m pybind_backend.run --data "$DATA512" --out "$OUTDIR/pybind_output_gpu_opt_512.hdf5" \
  --mode gpu-opt --epochs "$EPOCHS" --samples 512 \
  2>&1 | tee "$OUTDIR/pybind_gpu_opt_s1_512.log"

echo ""
echo "=== $(date) : [3/3] pybind 512^3 gpu-opt OSEM S=2, ${EPOCHS} epochs ==="
python3 -m pybind_backend.run --data "$DATA512" --out "$OUTDIR/pybind_output_gpu_opt_512_s2.hdf5" \
  --mode gpu-opt --subsets 2 --epochs "$EPOCHS" --samples 512 \
  2>&1 | tee "$OUTDIR/pybind_gpu_opt_s2_512.log"

echo "$(date)" > ct_kalefull_DONE
echo "$OUTDIR" >> ct_kalefull_DONE
echo ""
echo "=== $(date) : everything complete. Results in $OUTDIR/ ==="
