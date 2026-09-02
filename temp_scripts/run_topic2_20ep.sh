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
# Waits for a marker file before starting -- which one is chosen via
# MARKER, settable as an env var at launch time (default:
# ct512_DONE, the 512^3 100-epoch run's safe-completion sentinel,
# written after gpu-opt S=2 but BEFORE the untested/may-fail S=3
# attempt -- deliberately not waiting on ct512_ALL_DONE, since there's
# no reason to hold Topic_2 hostage to an experiment expected to fail).
# For the original 256^3-run-then-Topic_2 ordering, launch with
# MARKER=ct100_cpu_done.marker instead. Survives an SSH drop -- launch
# with (matching whichever run this should wait behind):
#
#   cd ~/CTreconstruction_GPU
#   nohup env MARKER=ct512_DONE bash temp_scripts/run_topic2_20ep.sh > ct_topic2_run.log 2>&1 &
#   disown
#
# Then check progress any time with:
#   tail -f ct_topic2_run.log
set -e
cd "$(dirname "$0")/.."

MARKER="${MARKER:-ct512_DONE}"
echo "=== $(date) : waiting for $MARKER ==="
while [ ! -f "$MARKER" ]; do
    sleep 30
done
echo "=== $(date) : $MARKER found (written $(cat "$MARKER" | head -1)), starting Topic_2 ==="

echo ""
echo "=== $(date) : Topic_2_CTreconstruction.py, 20 epochs, sample_ratio=1 ==="
python3 python/Topic_2_CTreconstruction.py 2>&1 | tee ct_topic2_output.log

echo ""
echo "=== $(date) : Topic_2 run complete ==="
echo "$(date)" > ct_topic2_DONE
