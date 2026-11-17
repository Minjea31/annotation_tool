from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from autorace_lane_lite.configuration import image_is_fresh, load_config, validate_config
from autorace_lane_lite.pipeline import LanePipeline
from autorace_lane_lite.tracking import LaneTracker

CONFIG = Path(__file__).resolve().parents[1] / 'config/lane.yaml'


def config():
    result = load_config(CONFIG)
    result['bev_source'] = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    return result


def mask_at(*positions):
    mask = np.zeros((240, 320), np.uint8)
    for x in positions:
        cv2.line(mask, (int(x * 319), 0), (int(x * 319), 239), 1, 5)
    return mask


class TrackingTests(unittest.TestCase):
    def test_centered_pair_and_single_boundary(self):
        for positions in ((0.275, 0.725), (0.275,), (0.725,)):
            estimate = LaneTracker(config()).update(mask_at(*positions))
            self.assertTrue(estimate.valid)
            self.assertLess(abs(estimate.steering), 0.015)

    def test_ackermann_sign_and_saturation(self):
        left = LaneTracker(config()).update(mask_at(0.175, 0.625))
        right = LaneTracker(config()).update(mask_at(0.375, 0.825))
        self.assertGreater(left.steering, 0)
        self.assertLess(right.steering, 0)
        self.assertLessEqual(abs(left.steering), config()['max_steering_rad'])

    def test_empty_does_not_reuse_last_command(self):
        tracker = LaneTracker(config())
        self.assertTrue(tracker.update(mask_at(0.275, 0.725)).valid)
        estimate = tracker.update(mask_at())
        self.assertFalse(estimate.valid)
        self.assertEqual((estimate.speed, estimate.steering), (0, 0))
        self.assertIsNone(tracker.previous)

    def test_fork_uses_explicit_fresh_direction(self):
        left = LaneTracker(config()).update(mask_at(0.1, 0.5, 0.9), branch_right=False)
        right = LaneTracker(config()).update(mask_at(0.1, 0.5, 0.9), branch_right=True)
        self.assertTrue(left.branch and right.branch)
        self.assertGreater(left.steering, 0)
        self.assertLess(right.steering, 0)

    def test_pipeline_load_failure_falls_back_without_torch(self):
        settings = config()
        settings.update(backend='tensorrt', model_path='/missing.engine')
        with patch('autorace_lane_lite.pipeline.create_backend', side_effect=RuntimeError('missing model')):
            pipeline = LanePipeline(settings)
            image = np.zeros((240, 320, 3), np.uint8)
            cv2.line(image, (87, 0), (87, 239), (0, 165, 255), 5)
            estimate, mask, stats = pipeline.process(image)
            self.assertTrue(estimate.valid)
            self.assertEqual(stats['backend'], 'opencv_fallback')
            self.assertEqual(mask.dtype, np.uint8)
            pipeline.close()

    def test_strict_backend_does_not_hide_load_failure(self):
        settings = config()
        settings.update(backend='tensorrt', model_path='/missing.engine', fallback_enabled=False)
        with patch('autorace_lane_lite.pipeline.create_backend', side_effect=RuntimeError('missing model')):
            with self.assertRaises(RuntimeError):
                LanePipeline(settings)

    def test_image_age_checks_ros_and_monotonic_time(self):
        settings = config()
        self.assertTrue(image_is_fresh(10, 10.1, 20, 20.1, settings))
        self.assertFalse(image_is_fresh(10, 10.3, 20, 20.1, settings))
        self.assertFalse(image_is_fresh(11, 10, 20, 20.1, settings))
        self.assertFalse(image_is_fresh(10, 10.1, 20, 21, settings))
        self.assertFalse(image_is_fresh(10, 10.1, 21, 20, settings))

    def test_invalid_resolution_rejected(self):
        settings = config()
        settings['input_width'] = 320
        with self.assertRaises(ValueError):
            validate_config(settings)


if __name__ == '__main__':
    unittest.main()
