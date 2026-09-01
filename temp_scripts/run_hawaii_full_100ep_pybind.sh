#!/bin/bash
# Fills the remaining gaps on pool15-01 (AMD Hawaii PRO) as of today's
# mclk-DVFS mitigation work (FP_BUFFER_SKIP_SLAB_FINISH, see README
# "gpu-buf run-to-run variance"):
#
#   0. CLI, gpu-buf ONLY, both resolutions, 100 epochs, WITH today's
#      mitigation on. The existing CLI 100-epoch set on the
#      pool15-results branch (2026-08-26 checkpoints) predates this
#      session's work -- its gpu-buf numbers reflect the unmitigated
#      baseline, not current code. cpu/gpu-img/gpu-opt are untouched by
#      today's changes and are NOT re-run (still valid as-is).
#   1. pybind, 256^3, all modes, 100 epochs -- Hawaii never had this
#      (kale does, see run_kale_full_100ep.sh, same structure, mirrored
#      here for the other machine).
#   2. pybind, 512^3, all modes, 100 epochs -- same gap.
#
# Two hardware targets are being compared in this investigation (kale =
# NVIDIA GTX 680, this = AMD Hawaii PRO) -- this script only covers the
# Hawaii side; run_kale_full_100ep.sh already covers kale and does not
# need today's mitigation (gpu-buf never showed this variance there).
#
# gpu-opt OSEM subset counts: 256^3 uses S=5 (same as kale, plenty of
# headroom at that size, no VRAM concern). 512^3 uses S=2 -- NOTE: unlike
# the header comment's original draft claimed, there is no actually-
# verified Hawaii-specific precedent for this in the repo (the "confirmed
# feasible" language in run_512_100ep.sh/run_kale_full_100ep.sh refers to
# the GTX 680's 4037 MiB limit, not this card's). Hawaii's VRAM budget may
# differ from kale's -- if this pybind S=2 512^3 run throws the same
# device-memory RuntimeError kale hit for a higher subset count (see
# pybind_backend's own VRAM pre-check), that is expected/handled
# (pipefail below stops the script cleanly rather than continuing past
# it), not a sign of a broken build. Reduce to --subsets 1 and rerun just
# that config if S=2 doesn't fit here.
#
# gpu-buf note: 100-epoch gpu-buf runs on THIS hardware are subject to
# the mclk-DVFS variance investigated separately (see README "gpu-buf
# run-to-run variance" and docs/performance-history.md's re-diagnosis
# section) -- expect run-to-run timing variance here that kale's GTX 680
# numbers never show. Not a bug; the correctness/MSE numbers are
# unaffected, only wall-clock time varies. FP_BUFFER_SKIP_SLAB_FINISH=1
# is set below for every gpu-buf run (CLI and pybind) since it's the
# documented mitigation (~29% faster mean) -- pass
# FP_BUFFER_SKIP_SLAB_FINISH=0 in the environment before this script if
# you specifically want unmitigated baseline numbers instead.
#
# NAMING: ct_hawaiifull_ prefix throughout, mirrors kale's ct_kalefull_
# collision-safety rationale.
#
# Launch inside tmux (this is a long run -- 2 CLI configs + 8 pybind
# configs, several of them 512^3):
#   tmux new -s hawaiifull
#   cd ~/CTreconstruction_GPU_pool15
#   bash temp_scripts/run_hawaii_full_100ep_pybind.sh 2>&1 | tee ct_hawaiifull_run.log
#   (detach: Ctrl+B then D; reattach: tmux attach -t hawaiifull)
set -e
set -o pipefail  # without this, `cmd | tee log` masks cmd's exit code with
                  # tee's -- a crashed run (e.g. gpu-opt S>=3 VRAM error,
                  # seen on kale for this exact config) would otherwise be
                  # silently swallowed and the script would continue as if
                  # nothing failed.
cd "$(dirname "$0")/.."

DATA256=/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5
DATA512=/lgrp/edu-2026-1-gpulab/proj_512_75.hdf5
EPOCHS=100
OUTDIR=results_hawaiifull_100ep_$(date +%Y%m%d_%H%M%S)
SKIP_FINISH="${FP_BUFFER_SKIP_SLAB_FINISH:-1}"

echo "=== $(date) : pre-flight ==="
if command -v klist >/dev/null 2>&1; then
  if ! klist -s 2>/dev/null; then
    echo "WARNING: no valid Kerberos ticket. kinit -R first if this fails."
    sleep 5
  else
    echo "kerberos ticket: OK"
  fi
fi
AVAIL_KB=$(df -Pk . | tail -1 | awk '{print $4}')
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
echo "disk available here: ${AVAIL_GB}GB"
if [ "$AVAIL_GB" -lt 4 ]; then
  # 2 CLI .hdf5 (256^3 ~65MB + 512^3 ~513MB) + 8 pybind .hdf5 (4x256^3
  # ~65MB + 4x512^3 ~513MB) = ~2.9GB of output alone, before logs and the
  # CLI build -- 4GB free is the minimum with any margin.
  echo "ERROR: less than 4GB free -- this run's outputs alone are ~2.9GB,"
  echo "  likely to fail partway through."
  echo "  Run ./temp_scripts/cleanup_before_run.sh first, or free space manually."
  exit 1
fi
echo "FP_BUFFER_SKIP_SLAB_FINISH=$SKIP_FINISH for gpu-buf runs below."
echo "=== pre-flight done ==="
echo ""

mkdir -p "$OUTDIR"

echo "=== $(date) : building CLI (build/ct_recon) ==="
make clean && make
ls -la build/ct_recon
# pybind extension is JIT-built by torch's load() on first import below
# (pybind_backend/backend.py) -- no separate pre-build step needed, same
# as run_kale_full_100ep.sh.

# ── Part 0: CLI, gpu-buf ONLY, both resolutions, WITH today's mitigation ─
# cpu/gpu-img/gpu-opt are untouched by FP_BUFFER_SKIP_SLAB_FINISH and are
# NOT re-run -- the existing pool15-results 100-epoch numbers for those
# modes are still current.
echo ""
echo "=== $(date) : [0/2] CLI 256^3 gpu-buf, ${EPOCHS} epochs (mitigated) ==="
FP_BUFFER_SKIP_SLAB_FINISH="$SKIP_FINISH" build/ct_recon --data "$DATA256" \
  --out "$OUTDIR/cli_output_gpu_buf_256_mitigated.hdf5" \
  --mode gpu-buf --epochs "$EPOCHS" --kernels kernels \
  2>&1 | tee "$OUTDIR/cli_gpu_buf_256_mitigated.log"

echo ""
echo "=== $(date) : [0/2] CLI 512^3 gpu-buf, ${EPOCHS} epochs (mitigated; expect run-to-run variance -- see header) ==="
FP_BUFFER_SKIP_SLAB_FINISH="$SKIP_FINISH" build/ct_recon --data "$DATA512" \
  --out "$OUTDIR/cli_output_gpu_buf_512_mitigated.hdf5" \
  --mode gpu-buf --epochs "$EPOCHS" --samples 512 --kernels kernels \
  2>&1 | tee "$OUTDIR/cli_gpu_buf_512_mitigated.log"

echo "$(date)" > ct_hawaiifull_cli_gpubuf_done.marker

rm -rf ~/.cache/torch_extensions/ct_recon-*

# ── Part 1: pybind, 256^3, all modes ───────────────────────────────────
echo ""
echo "=== $(date) : [1/2] pybind 256^3 cpu, ${EPOCHS} epochs ==="
python3 -m pybind_backend.run --data "$DATA256" --out "$OUTDIR/pybind_output_cpu_256.hdf5" \
  --mode cpu --epochs "$EPOCHS" \
  2>&1 | tee "$OUTDIR/pybind_cpu_256.log"

echo ""
echo "=== $(date) : [1/2] pybind 256^3 gpu-buf, ${EPOCHS} epochs ==="
FP_BUFFER_SKIP_SLAB_FINISH="$SKIP_FINISH" python3 -m pybind_backend.run \
  --data "$DATA256" --out "$OUTDIR/pybind_output_gpu_buf_256.hdf5" \
  --mode gpu-buf --epochs "$EPOCHS" \
  2>&1 | tee "$OUTDIR/pybind_gpu_buf_256.log"

echo ""
echo "=== $(date) : [1/2] pybind 256^3 gpu-img, ${EPOCHS} epochs ==="
python3 -m pybind_backend.run --data "$DATA256" --out "$OUTDIR/pybind_output_gpu_img_256.hdf5" \
  --mode gpu-img --epochs "$EPOCHS" \
  2>&1 | tee "$OUTDIR/pybind_gpu_img_256.log"

echo ""
echo "=== $(date) : [1/2] pybind 256^3 gpu-opt (S=1), ${EPOCHS} epochs ==="
python3 -m pybind_backend.run --data "$DATA256" --out "$OUTDIR/pybind_output_gpu_opt_256.hdf5" \
  --mode gpu-opt --epochs "$EPOCHS" \
  2>&1 | tee "$OUTDIR/pybind_gpu_opt_s1_256.log"

echo ""
echo "=== $(date) : [1/2] pybind 256^3 gpu-opt OSEM S=5, ${EPOCHS} epochs ==="
python3 -m pybind_backend.run --data "$DATA256" --out "$OUTDIR/pybind_output_gpu_opt_256_s5.hdf5" \
  --mode gpu-opt --subsets 5 --epochs "$EPOCHS" \
  2>&1 | tee "$OUTDIR/pybind_gpu_opt_s5_256.log"

echo "$(date)" > ct_hawaiifull_pybind256_done.marker

# ── Part 2: pybind, 512^3, all modes ───────────────────────────────────
echo ""
echo "=== $(date) : [2/2] pybind 512^3 cpu, ${EPOCHS} epochs (expect ~55-60 min) ==="
python3 -m pybind_backend.run --data "$DATA512" --out "$OUTDIR/pybind_output_cpu_512.hdf5" \
  --mode cpu --epochs "$EPOCHS" --samples 512 \
  2>&1 | tee "$OUTDIR/pybind_cpu_512.log"

echo ""
echo "=== $(date) : [2/2] pybind 512^3 gpu-buf, ${EPOCHS} epochs (expect run-to-run variance -- see header) ==="
FP_BUFFER_SKIP_SLAB_FINISH="$SKIP_FINISH" python3 -m pybind_backend.run \
  --data "$DATA512" --out "$OUTDIR/pybind_output_gpu_buf_512.hdf5" \
  --mode gpu-buf --epochs "$EPOCHS" --samples 512 \
  2>&1 | tee "$OUTDIR/pybind_gpu_buf_512.log"

echo ""
echo "=== $(date) : [2/2] pybind 512^3 gpu-img, ${EPOCHS} epochs ==="
python3 -m pybind_backend.run --data "$DATA512" --out "$OUTDIR/pybind_output_gpu_img_512.hdf5" \
  --mode gpu-img --epochs "$EPOCHS" --samples 512 \
  2>&1 | tee "$OUTDIR/pybind_gpu_img_512.log"

echo ""
echo "=== $(date) : [2/2] pybind 512^3 gpu-opt (S=1), ${EPOCHS} epochs ==="
python3 -m pybind_backend.run --data "$DATA512" --out "$OUTDIR/pybind_output_gpu_opt_512.hdf5" \
  --mode gpu-opt --epochs "$EPOCHS" --samples 512 \
  2>&1 | tee "$OUTDIR/pybind_gpu_opt_s1_512.log"

echo ""
echo "=== $(date) : [2/2] pybind 512^3 gpu-opt OSEM S=2, ${EPOCHS} epochs (VRAM fit NOT independently confirmed on this card -- see header note; S>=3 not attempted) ==="
python3 -m pybind_backend.run --data "$DATA512" --out "$OUTDIR/pybind_output_gpu_opt_512_s2.hdf5" \
  --mode gpu-opt --subsets 2 --epochs "$EPOCHS" --samples 512 \
  2>&1 | tee "$OUTDIR/pybind_gpu_opt_s2_512.log"

echo "$(date)" > ct_hawaiifull_DONE
echo "$OUTDIR" >> ct_hawaiifull_DONE
echo ""
echo "=== $(date) : everything complete. Results in $OUTDIR/ ==="
find "$OUTDIR" -name "*.log" | sort
