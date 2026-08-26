#!/bin/bash
# ARCHIVED: paths assume this script runs from repo root (its original
# location before the temp scripts/ cleanup move) -- "../python/X.py"
# paths need re-checking before rerunning from here.
#
# Runs Topic_2_CTreconstruction_512.py (the 512^3 variant of the
# course-provided Python reference, python/Topic_2_CTreconstruction_512.py)
# under a memory cap. Real risk here, not hypothetical: a prior attempt at
# 512^3 in this project OOM'd and, once, froze the entire machine badly
# enough to need a hard reset -- on AMD Hawaii PRO (pool15-01), which only
# has 15GB RAM and no swap (see README's session log). The memory cap below
# turns a repeat of that into a clean kill instead of a freeze -- it does
# NOT make the workload fit if it doesn't fit; if this script gets killed,
# that confirms it doesn't fit on this machine, not a script bug.
#
# NVIDIA GTX 680's machine (kale) has 188GB RAM, much safer, but 512^3
# timing there is still unmeasured -- run this first before committing to
# it on AMD Hawaii PRO.
#
# Run inside tmux so it survives a disconnect:
#   tmux new -s topic2_512
#   ./topic2_512.sh
#   (detach: Ctrl+B then D)
#
# MEM_CAP_GB: override the memory cap (default 8GB -- conservative, well
# under AMD Hawaii PRO's 15GB total so the OS/other processes aren't
# starved too). On kale (188GB RAM) you can raise this, e.g.
# MEM_CAP_GB=32 ./topic2_512.sh
set -e

MEM_CAP_GB=${MEM_CAP_GB:-8}
MEM_CAP_KB=$((MEM_CAP_GB * 1024 * 1024))

echo "=== pre-flight ==="
echo "memory cap: ${MEM_CAP_GB}GB (override with MEM_CAP_GB=N)"
if command -v free >/dev/null 2>&1; then
  free -g
fi
echo ""
echo "Starting Topic_2_CTreconstruction_512.py under this cap. If it gets"
echo "OOM-killed, that means 512^3 doesn't fit at this cap on this machine --"
echo "raise MEM_CAP_GB (if RAM allows) or treat 512^3 as infeasible here,"
echo "matching the earlier AMD Hawaii PRO finding."
echo ""

( ulimit -v "$MEM_CAP_KB"; python3 ../python/Topic_2_CTreconstruction_512.py ) 2>&1 | tee topic2_512.log

echo ""
echo "=== done (or killed -- check exit status above) ==="
