#!/usr/bin/env bash
# Build the upload bundle for a cloud training run (run on the laptop, from the repo root, in Git Bash):
#
#     bash tools/cloud/pack.sh [RUN_NAME] [SOURCE_RUN]      # defaults: cloud_a  runs/9x9_a
#
# Contents (~240 MB): the repo's files as they are now (tracked + new, not ignored), the source
# run's latest model + replay buffers + metrics as runs/RUN_NAME, and the frozen anchor c439.
# VERSION records `git describe --always --dirty`. Output: dist/RUN_NAME_bundle.tar.gz
set -euo pipefail
RUN=${1:-cloud_a}
SRC=${2:-runs/9x9_a}
cd "$(git rev-parse --show-toplevel)"
[ -f "configs/${RUN}.yaml" ] || { echo "configs/${RUN}.yaml not found" >&2; exit 1; }
STAGE=$(mktemp -d)
git ls-files -co --exclude-standard | grep -v '^dist/' | tar -cf - -T - | tar -xf - -C "$STAGE"
git describe --always --dirty > "$STAGE/VERSION"
mkdir -p "$STAGE/runs/$RUN" "$STAGE/runs/eval_night6"
for f in latest.pt buffer.npz val_buffer.npz metrics.csv; do cp "$SRC/$f" "$STAGE/runs/$RUN/"; done
cp runs/eval_night6/candidate_c439.pt "$STAGE/runs/eval_night6/"
mkdir -p dist
tar -czf "dist/${RUN}_bundle.tar.gz" -C "$STAGE" .
rm -rf "$STAGE"
echo "version $(git describe --always --dirty)"
ls -lh "dist/${RUN}_bundle.tar.gz"
