#!/usr/bin/env bash
# On the instance: self-play speed at several worker counts (~12 min), to pick selfplay_workers.
#
#     bash tools/cloud/bench.sh [RUN_NAME] [WORKER COUNTS...]   # default: cloud_a 1 4 8 12
set -euo pipefail
RUN=${1:-cloud_a}
shift || true
COUNTS=${*:-1 4 8 12}
cd /workspace/go-zero
python -u tools/sp_bench.py --config "configs/${RUN}.yaml" --model "runs/${RUN}/latest.pt" \
  --workers $COUNTS --seconds 90 --warmup 30 | tee "runs/${RUN}_bench.txt"
echo "laptop reference: ~3,500 evals/s single process (nights 6-7)"
