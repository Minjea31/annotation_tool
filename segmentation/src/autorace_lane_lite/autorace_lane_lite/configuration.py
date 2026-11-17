import math
from pathlib import Path
import yaml


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())['lane_node']['ros__parameters']
    validate_config(config)
    return config


def validate_config(config):
    if (config['input_width'], config['input_height']) not in ((640, 480), (512, 288)):
        raise ValueError('Supported model inputs: 640x480 or 512x288')
    if config['backend'] not in ('auto', 'opencv', 'torch', 'onnx', 'tensorrt'):
        raise ValueError('Unknown inference backend')
    for key in ('bev_width', 'bev_height', 'bev_lateral_m', 'bev_forward_m', 'lookahead_m',
                'wheelbase_m', 'max_steering_rad', 'max_center_offset_m', 'process_hz',
                'max_image_age_seconds', 'retry_seconds', 'window_min_pixels'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if config['bev_width'] < 32 or config['bev_height'] < 32:
        raise ValueError('BEV dimensions must be >=32')
    if not 1 <= config['min_windows'] <= 9:
        raise ValueError('min_windows must be 1..9')
    for key in ('lane_width_ratio', 'window_margin_ratio', 'tracking_alpha', 'confidence_threshold'):
        if not 0 < config[key] <= 1:
            raise ValueError(f'{key} must be in (0,1]')
    for key in ('speed_mps', 'curve_slowdown', 'heading_gain', 'future_tolerance_seconds', 'sign_ttl_seconds'):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if config['lookahead_m'] > config['bev_forward_m']:
        raise ValueError('lookahead_m exceeds the BEV range')


def image_is_fresh(stamp, now, received_monotonic, now_monotonic, config):
    age = now - stamp
    elapsed = now_monotonic - received_monotonic
    return (-config['future_tolerance_seconds'] <= age <= config['max_image_age_seconds']
            and 0 <= elapsed <= config['max_image_age_seconds'])
