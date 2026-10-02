#!/usr/bin/env bash
# On the instance: start training in the background (survives closing the SSH session).
#
#     bash tools/cloud/train.sh RUN_NAME WORKERS HOURS
#
# Sets loop.selfplay_workers in configs/RUN_NAME.yaml, then runs the normal loop on runs/RUN_NAME.
# HOURS must leave money for pulling the results (see NOTES "Cloud run"): the instance stops when
# the account balance runs out, and an unpulled run would be lost with it.
set -euo pipefail
RUN=$1; WORKERS=$2; HOURS=$3
cd /workspace/go-zero
sed -i -E "s/^(  selfplay_workers:).*/\1 ${WORKERS}/" "configs/${RUN}.yaml"
grep -nE "selfplay_workers|value_q_weight|lr:" "configs/${RUN}.yaml"
nohup python -u -m gozero.loop --config "configs/${RUN}.yaml" --run "runs/${RUN}" --hours "$HOURS" \
  >> "runs/${RUN}.log" 2>&1 &
echo "started pid $! ; follow with: tail -f /workspace/go-zero/runs/${RUN}.log"
