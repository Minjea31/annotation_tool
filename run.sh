#!/usr/bin/env bash
set -euo pipefail
annotation_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
annotation_python="${ANNOTATION_PYTHON:-${LANE_REVIEW_PYTHON:-python3}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec "$annotation_python" "$annotation_dir/server.py"
