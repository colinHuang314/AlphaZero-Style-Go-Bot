#!/usr/bin/env bash
# On the cloud instance (Linux, PyTorch image), after uploading the bundle to /workspace:
#
#     bash setup.sh [RUN_NAME]        # unpacks to /workspace/go-zero, installs deps, runs the tests
set -euo pipefail
RUN=${1:-cloud_a}
mkdir -p /workspace/go-zero
tar -xzf "/workspace/${RUN}_bundle.tar.gz" -C /workspace/go-zero
cd /workspace/go-zero
python -m pip install -q numba pyyaml pytest
python - <<'EOF'
import os, torch
print("python ok | torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", "| cpus", os.cpu_count())
EOF
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
free -g | head -2
python -m pytest -q tests
echo "setup done: next  bash tools/cloud/bench.sh $RUN"
