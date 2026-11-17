"""Offline image inference, BEV and optional overlay; no ROS required."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401
from autorace_lane_lite.configuration import load_config, validate_config
from autorace_lane_lite.pipeline import LanePipeline

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'src/autorace_lane_lite/config/lane.yaml'


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--backend', choices=['torch', 'onnx', 'tensorrt', 'opencv'], required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--size', choices=['640x480', '512x288'], default='512x288')
    parser.add_argument('--allow-fallback', action='store_true', help='Default: fail if the selected neural backend fails')
    parser.add_argument('--right-lane', action='store_true')
    return parser


def configuration(args):
    config = load_config(args.config)
    config.update(backend=args.backend, model_path=str(args.model) if args.model else '', device=args.device,
                  fallback_enabled=args.allow_fallback, input_width=int(args.size.split('x')[0]),
                  input_height=int(args.size.split('x')[1]))
    if args.backend != 'opencv' and args.model is None:
        raise ValueError('--model is required for neural inference')
    validate_config(config)
    return config


def main():
    parser = arguments(__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new directory')
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error('Cannot read image')
    pipeline = LanePipeline(configuration(args))
    try:
        estimate, mask, stats = pipeline.process(image, args.right_lane)
        args.output.mkdir(parents=True)
        colors = np.array([[0, 0, 0], [0, 128, 255], [255, 255, 255]], np.uint8)
        overlay = image.copy()
        foreground = mask > 0
        overlay[foreground] = (0.5 * overlay[foreground] + 0.5 * colors[mask[foreground]]).astype(np.uint8)
        cv2.imwrite(str(args.output / 'mask.png'), mask)
        cv2.imwrite(str(args.output / 'overlay.jpg'), overlay)
        cv2.imwrite(str(args.output / 'bev.png'), colors[pipeline.tracker.bev])
        result = {'estimate': vars(estimate), **stats}
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
    finally:
        pipeline.close()


if __name__ == '__main__':
    main()
