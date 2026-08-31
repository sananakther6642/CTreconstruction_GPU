#!/bin/bash
# Runs python/Topic_2_CTreconstruction.py (the course's Python reference
# implementation, hardcoded to 20 epochs / sample_ratio=1 at
# python/Topic_2_CTreconstruction.py:240,236 -- already matches the
# report's methodology, no arguments needed). Expected runtime: very
# long (~20+ hours per earlier session notes at this resolution).
#
# Waits for run_100ep_final.sh's CPU run to finish first (its own
# sentinel, cpu_done.marker) before starting -- run_100ep_final.sh's CPU
# pass is done early (before its GPU modes), so this doesn't wait for
# the GPU runs. Survives an SSH drop -- launch with:
#
#   cd ~/CTreconstruction_GPU
#   nohup bash temp_scripts/run_topic2_20ep.sh > run_topic2_20ep.log 2>&1 &
#   disown
#
# Then check progress any time with:
#   tail -f run_topic2_20ep.log
set -e
cd "$(dirname "$0")/.."

MARKER=cpu_done.marker
echo "=== $(date) : waiting for $MARKER (run_100ep_final.sh's CPU pass) ==="
while [ ! -f "$MARKER" ]; do
    sleep 30
done
echo "=== $(date) : $MARKER found (written $(cat "$MARKER")), starting Topic_2 ==="

echo ""
echo "=== $(date) : Topic_2_CTreconstruction.py, 20 epochs, sample_ratio=1 ==="
python3 python/Topic_2_CTreconstruction.py 2>&1 | tee topic2_20ep.log

echo ""
echo "=== $(date) : Topic_2 run complete ==="
echo "$(date)" > run_topic2_20ep.DONE
