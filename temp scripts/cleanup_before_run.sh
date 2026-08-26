#!/bin/bash
# Check disk space and clear stale .hdf5 output files before a big run.
# Safe: only deletes output_*.hdf5 / *_100ep.hdf5 / conv-sweep tmp outputs --
# never touches src/, kernels/, sessions/, or anything git-tracked.
set -e

echo "=== disk space before ==="
df -h ~
echo ""

echo "=== largest .hdf5 files in repo (candidates for cleanup) ==="
du -sh *.hdf5 2>/dev/null | sort -rh | head -20
echo ""

echo "=== deleting known-stale one-off outputs ==="
rm -fv output_osem_*.hdf5 output_S*_100ep.hdf5 output_sweep_*.hdf5 \
       output_gpu_opt_s*.hdf5 output_c3_*.hdf5 output_c5_*.hdf5 \
       output_2x2_*.hdf5 output_fdk.hdf5 output_diag.hdf5 2>/dev/null || true

echo ""
echo "=== /tmp sweep outputs from earlier (safe to clear, not report inputs) ==="
rm -fv /tmp/osem_s*.hdf5 /tmp/mlem_*.hdf5 2>/dev/null || true

echo ""
echo "=== KEEPING (needed for current work) ==="
echo "  output_cpu.hdf5, output_gpu_{buf,img,opt}.hdf5       (256^3 current)"
echo "  output_cpu_512.hdf5, output_gpu_{buf,img,opt}_512.hdf5 (512^3 current)"
echo "  conv_csv/          (kale convergence sweep results)"
echo "  pool15/             (pool15-01 results, once generated)"
echo ""

echo "=== disk space after ==="
df -h ~
