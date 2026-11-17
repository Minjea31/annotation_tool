"""Measure decode+segmentation+BEV+tracking+control, not network-only FPS."""
import json
from pathlib import Path
import time

import cv2
import numpy as np

from infer import arguments, configuration
from autorace_lane_lite.pipeline import LanePipeline


def main():
    parser = arguments(__doc__)
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--frames', type=int, default=500)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--yolo-running', action='store_true', help='Record that YOU separately started YOLO26n')
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0:
        parser.error('Invalid frame counts')
    if args.output.exists():
        parser.error('Output exists; choose a new path')
    paths = sorted(p for p in args.images.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if not paths:
        parser.error('No camera images found')
    # Preload a bounded set of COMPRESSED bytes so timing excludes disk I/O.
    selection = [paths[i] for i in np.linspace(0, len(paths) - 1, min(len(paths), 128), dtype=int)]
    buffers = [np.frombuffer(p.read_bytes(), np.uint8) for p in selection]
    pipeline = LanePipeline(configuration(args))
    duration, modes, valid, valid_neural = [], {}, 0, 0
    try:
        for index in range(args.warmup + args.frames):
            start = time.perf_counter()
            image = cv2.imdecode(buffers[index % len(buffers)], cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('Invalid compressed camera image')
            estimate, _, stats = pipeline.process(image, args.right_lane)
            elapsed = (time.perf_counter() - start) * 1000
            if index >= args.warmup:
                duration.append(elapsed)
                mode = stats['backend']
                modes[mode] = modes.get(mode, 0) + 1
                valid += int(estimate.valid)
                valid_neural += int(estimate.valid and mode in ('torch', 'onnx', 'tensorrt'))
    finally:
        pipeline.close()
    report = {'size': args.size, 'requested_backend': args.backend, 'frames': args.frames,
              'image_selection': [str(path.relative_to(args.images)) for path in selection],
              'mean_ms': float(np.mean(duration)), 'p50_ms': float(np.percentile(duration, 50)),
              'p95_ms': float(np.percentile(duration, 95)), 'throughput_fps': 1000.0 / float(np.mean(duration)),
              'valid_lane_fraction': valid / args.frames, 'valid_neural_fraction': valid_neural / args.frames,
              'backend_frames': modes, 'yolo_running_user_declared': args.yolo_running,
              'includes': 'CPU decode + preprocess + inference/transfers + mask + BEV + tracking + control',
              'excludes': 'camera transport, ROS scheduling/publication and disk reads',
              'note': 'Speed is not accuracy. Compare held-out lane IoU and driving geometry separately.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
