#!/usr/bin/env bash
set -euo pipefail
review_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
review_python="${LANE_REVIEW_PYTHON:-python3}"
if [[ ! -x "$review_python" ]]; then
  review_python="python3"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec "$review_python" "$review_dir/app.py"
