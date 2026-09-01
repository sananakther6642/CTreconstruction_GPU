#!/bin/bash
# FP_TEX_EXACT validation on AMD Hawaii PRO (pool15-01), 100 epochs.
#
# WHY
# FP_TEX_EXACT (kernels/fp_image.cl) closes the gpu-img/gpu-opt precision
# gap by keeping the 3D texture cache while replacing the sampler's
# fixed-function blend with an IEEE float32 one. Measured on kale
# (GTX 680, 100 epochs, MSE vs CPU):
#     256^3   1.128e-07 -> 2.524e-09  (45x),  14.65s -> 21.34s
#     512^3   1.232e-09 -> 5.528e-10  (2.2x)
# It has NEVER been run on AMD. The kernel is shared by both machines and
# AMD's OpenCL compiler is stricter, so this confirms it builds, runs, and
# reproduces the accuracy gain on Hawaii rather than assuming it transfers.
#
# Also fills two gaps the 2026-09-01 sweep left:
#   - CLI gpu-opt at both resolutions (skipped there as "unchanged", which
#     predates this precision work)
#   - gpu-opt timing under FP_TEX_EXACT (its MSE is measured on kale, its
#     runtime never was)
#
# RUN THE 3-EPOCH SMOKE TEST FIRST, INTERACTIVELY. If the kernel fails to
# build on AMD, an unattended run wastes the whole slot:
#     ssh pool15-01
#     cd ~/CTreconstruction_GPU_pool15 && git pull && make clean && make
#     FP_TEX_EXACT=1 build/ct_recon --data /lgrp/edu-2026-1-gpulab/proj_256_75.hdf5 \
#       --out /tmp/t.hdf5 --mode gpu-img --epochs 3 --kernels kernels
# Confirm it prints "OpenCL device: Hawaii" and sane epoch times, THEN:
#
#     tmux new -s hawaiiprec
#     cd ~/CTreconstruction_GPU_pool15
#     bash temp_scripts/run_hawaii_precision_100ep.sh 2>&1 | tee ct_hawaiiprec_run.log
#     (detach: Ctrl+B then D)
#
# Runtime ~2.5h, dominated by the two CPU reference runs. Disk ~1.5GB.
#
# NOTE ON gpu-buf: FP_TEX_EXACT does not touch gpu-buf (it has no
# fp_image), so gpu-buf here is a CONTROL. It should reproduce the known
# Hawaii figures; if it does not, the CPU references are wrong and nothing
# else in this run is interpretable. Expect gpu-buf 512^3 to be slow and
# erratic on this machine -- that is the documented mclk-DVFS variance,
# not a fault of this change.
set -e
set -o pipefail   # else `cmd | tee` masks a failing cmd's exit status
cd "$(dirname "$0")/.."

echo "=== $(date) : pre-flight ==="
if command -v klist >/dev/null 2>&1; then
  klist -s 2>/dev/null && echo "kerberos: OK" || {
    echo "WARNING: no valid Kerberos ticket -- NFS home access will fail"
    echo "  mid-run and this script will die. Run 'kinit akthersn' FIRST."
    echo "  (kinit -R does not work once a ticket has already expired.)"
    echo "  Continuing in 10s -- Ctrl+C to abort."
    sleep 10
  }
fi
AVAIL_GB=$(( $(df -Pk . | tail -1 | awk '{print $4}') / 1024 / 1024 ))
echo "disk available: ${AVAIL_GB}GB"
if [ "$AVAIL_GB" -lt 4 ]; then
  echo "ERROR: under 4GB free; this run writes ~1.5GB of volumes plus logs."
  exit 1
fi

DATA256=/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5
DATA512=/lgrp/edu-2026-1-gpulab/proj_512_75.hdf5
EPOCHS=100
OUT=results_hawaiiprec_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT/base256" "$OUT/exact256" "$OUT/base512" "$OUT/exact512" "$OUT/logs"

echo "=== $(date) : build ==="
make clean && make
ls -l build/ct_recon

run() {  # run <dir> <tag> <mode> <data> <epochs> <exact 0|1> [extra...]
  local d="$1" tag="$2" mode="$3" data="$4" ep="$5" exact="$6"; shift 6
  echo ""
  echo "--- $(date +%H:%M:%S)  $tag  mode=$mode epochs=$ep FP_TEX_EXACT=$exact ---"
  if [ "$exact" = "1" ]; then export FP_TEX_EXACT=1; else unset FP_TEX_EXACT; fi
  build/ct_recon --data "$data" --out "$d/output_${tag}.hdf5" \
    --mode "$mode" --epochs "$ep" --kernels kernels "$@" \
    2>&1 | tee "$OUT/logs/${tag}_$(basename "$d").log"
  unset FP_TEX_EXACT
}

# ── 256^3 ───────────────────────────────────────────────────────────────
# One CPU reference serves both arms: FP_TEX_EXACT never affects cpu mode,
# so it is generated once and symlinked, saving a ~4.5 min run.
echo ""
echo "######## $(date) : 256^3 ########"
run "$OUT/base256" cpu cpu "$DATA256" $EPOCHS 0
ln -sf "$PWD/$OUT/base256/output_cpu.hdf5" "$OUT/exact256/output_cpu.hdf5"

for M in gpu-buf gpu-img gpu-opt; do
  t=$(echo "$M" | tr '-' '_')
  run "$OUT/base256"  "$t" "$M" "$DATA256" $EPOCHS 0
done
# gpu-buf has no fp_image, so it is unaffected -- symlink rather than re-run.
ln -sf "$PWD/$OUT/base256/output_gpu_buf.hdf5" "$OUT/exact256/output_gpu_buf.hdf5"
for M in gpu-img gpu-opt; do
  t=$(echo "$M" | tr '-' '_')
  run "$OUT/exact256" "$t" "$M" "$DATA256" $EPOCHS 1
done

echo ""
echo "=== 256^3 baseline ==="; python3 python/validate.py --dir "$OUT/base256"  2>&1 | tee "$OUT/logs/validate_base256.txt"
echo "=== 256^3 FP_TEX_EXACT ==="; python3 python/validate.py --dir "$OUT/exact256" 2>&1 | tee "$OUT/logs/validate_exact256.txt"
echo "$(date)" > ct_hawaiiprec_256_done.marker

# ── 512^3 ───────────────────────────────────────────────────────────────
echo ""
echo "######## $(date) : 512^3  (cpu ~1h) ########"
OMP_NUM_THREADS=$(nproc) OMP_PROC_BIND=close OMP_PLACES=cores \
  run "$OUT/base512" cpu_512 cpu "$DATA512" $EPOCHS 0 --samples 512
ln -sf "$PWD/$OUT/base512/output_cpu_512.hdf5" "$OUT/exact512/output_cpu_512.hdf5"

for M in gpu-buf gpu-img gpu-opt; do
  t=$(echo "$M" | tr '-' '_')_512
  run "$OUT/base512"  "$t" "$M" "$DATA512" $EPOCHS 0 --samples 512
done
ln -sf "$PWD/$OUT/base512/output_gpu_buf_512.hdf5" "$OUT/exact512/output_gpu_buf_512.hdf5"
for M in gpu-img gpu-opt; do
  t=$(echo "$M" | tr '-' '_')_512
  run "$OUT/exact512" "$t" "$M" "$DATA512" $EPOCHS 1 --samples 512
done

echo ""
echo "=== 512^3 baseline ==="; python3 python/validate.py 512 --dir "$OUT/base512"  2>&1 | tee "$OUT/logs/validate_base512.txt"
echo "=== 512^3 FP_TEX_EXACT ==="; python3 python/validate.py 512 --dir "$OUT/exact512" 2>&1 | tee "$OUT/logs/validate_exact512.txt"

echo "$(date)" > ct_hawaiiprec_DONE
echo "$OUT" >> ct_hawaiiprec_DONE

echo ""
echo "=== $(date) : DONE.  $OUT/ ==="
echo ""
echo "=== timings ==="
grep -H "GPU time\|GPU-opt time\|CPU time" "$OUT"/logs/*.log
echo ""
echo "HOW TO READ IT:"
echo "  1. gpu-buf is the CONTROL (FP_TEX_EXACT cannot affect it). If its MSE"
echo "     does not match the known Hawaii figures, the CPU reference is wrong"
echo "     and nothing else here is interpretable."
echo "  2. Does FP_TEX_EXACT reproduce kale's gain on AMD? kale went"
echo "     1.128e-07 -> 2.524e-09 at 256^3 and 1.232e-09 -> 5.528e-10 at 512^3."
echo "     Hawaii's absolute numbers differ from kale's (its gpu-buf 256^3 MSE"
echo "     is ~5.6x looser, an open question predating this work), so compare"
echo "     the RATIO baseline->exact, not the raw values."
echo "  3. Time cost: kale paid 1.46x at 256^3. AMD may differ -- its texture"
echo "     cache and ALU balance are not the same."
echo "  4. gpu-buf 512^3 will likely be slow and erratic. That is the"
echo "     documented mclk-DVFS variance, unrelated to this change."
