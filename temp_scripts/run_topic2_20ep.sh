#!/bin/bash
# Runs python/Topic_2_CTreconstruction.py (the course's Python reference
# implementation, hardcoded to 20 epochs / sample_ratio=1 at
# python/Topic_2_CTreconstruction.py:240,236 -- already matches the
# report's methodology, no arguments needed). Expected runtime: very
# long (~20+ hours per earlier session notes at this resolution).
#
# NAMING: every file this script owns is prefixed ct_topic2_
# (ct_topic2_run.log, ct_topic2_DONE) -- see run_100ep_final.sh's
# comment for why (shared-machine collision safety + easy cleanup).
# The actual reference output goes wherever Topic_2_CTreconstruction.py
# itself writes it (unchanged from the script's own behavior).
#
# Waits for run_100ep_final.sh's CPU run to finish first (its sentinel,
# ct100_cpu_done.marker) before starting -- that script's CPU pass
# finishes early (before its GPU modes), so this doesn't wait for the
# GPU runs, only the CPU pass, to avoid the two contending for host CPU
# cycles. Survives an SSH drop -- launch with:
#
#   cd ~/CTreconstruction_GPU
#   nohup bash temp_scripts/run_topic2_20ep.sh > ct_topic2_run.log 2>&1 &
#   disown
#
# Then check progress any time with:
#   tail -f ct_topic2_run.log
set -e
cd "$(dirname "$0")/.."

MARKER=ct100_cpu_done.marker
echo "=== $(date) : waiting for $MARKER (run_100ep_final.sh's CPU pass) ==="
while [ ! -f "$MARKER" ]; do
    sleep 30
done
echo "=== $(date) : $MARKER found (written $(cat "$MARKER")), starting Topic_2 ==="

echo ""
echo "=== $(date) : Topic_2_CTreconstruction.py, 20 epochs, sample_ratio=1 ==="
python3 python/Topic_2_CTreconstruction.py 2>&1 | tee ct_topic2_output.log

echo ""
echo "=== $(date) : Topic_2 run complete ==="
echo "$(date)" > ct_topic2_DONE
