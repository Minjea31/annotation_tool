#!/usr/bin/env bash
set -euo pipefail
annotation_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
annotation_python="${ANNOTATION_PYTHON:-${LANE_REVIEW_PYTHON:-}}"
if [[ -z "$annotation_python" ]]; then
  for candidate in "$annotation_dir/.venv/bin/python" python3 "$HOME/anaconda3/envs/yolo/bin/python"; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import importlib.util; import sys; sys.exit(not all(importlib.util.find_spec(m) for m in ("fastapi", "uvicorn", "numpy", "PIL", "torch", "torchvision")))' 2>/dev/null; then
      annotation_python="$candidate"
      break
    fi
  done
fi
if [[ -z "$annotation_python" ]]; then
  echo '실행에 필요한 Python 패키지를 찾을 수 없습니다. README.md의 설치 절차를 진행하거나 ANNOTATION_PYTHON을 지정하세요.' >&2
  exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
echo "Annotation: http://127.0.0.1:${ANNOTATION_PORT:-8765} (Python: $annotation_python)"
exec "$annotation_python" "$annotation_dir/server.py"
