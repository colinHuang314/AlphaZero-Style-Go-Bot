#!/usr/bin/env bash
# On the laptop (Git Bash, repo root): copy a cloud run's results back.
#
#     bash tools/cloud/pull.sh HOST PORT [RUN_NAME] [--full]
#
# Default: latest model, metrics, log and bench result (~25 MB), safe to run any time as a backup.
# --full also fetches the replay buffers (~200 MB, needed to continue the run on the laptop) and the
# model snapshots saved every 10 cycles (~23 MB each). Uses the key ~/.ssh/vast_gozero.
set -euo pipefail
HOST=$1; PORT=$2; RUN=${3:-cloud_a}; FULL=${4:-}
KEY=~/.ssh/vast_gozero
R=/workspace/go-zero/runs
cd "$(git rev-parse --show-toplevel)"
mkdir -p "runs/$RUN/models"
SCP="scp -P $PORT -i $KEY -o StrictHostKeyChecking=accept-new"
$SCP "root@$HOST:$R/$RUN/latest.pt" "root@$HOST:$R/$RUN/metrics.csv" "runs/$RUN/"
$SCP "root@$HOST:$R/$RUN.log" "runs/" || true
$SCP "root@$HOST:$R/${RUN}_bench.txt" "runs/" 2>/dev/null || true
if [ "$FULL" = "--full" ]; then
  $SCP "root@$HOST:$R/$RUN/buffer.npz" "root@$HOST:$R/$RUN/val_buffer.npz" "runs/$RUN/"
  $SCP "root@$HOST:$R/$RUN/models/*" "runs/$RUN/models/" 2>/dev/null || true
fi
tail -n 3 "runs/$RUN.log" 2>/dev/null || true
ls -l "runs/$RUN"
