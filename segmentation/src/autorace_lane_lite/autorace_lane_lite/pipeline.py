import time
import cv2
import numpy as np

from .runtime import OpenCVBackend, create_backend, decode, preprocess
from .tracking import LaneTracker


class LanePipeline:
    def __init__(self, config):
        self.config = config
        self.classical = OpenCVBackend(config)
        self.tracker = LaneTracker(config)
        self.backend, self.error = None, ''
        self.retry_after = 0.0
        self._try_load()

    def _try_load(self):
        if self.config['backend'] == 'opencv':
            return
        if self.config['backend'] == 'auto' and not self.config['model_path']:
            self.error = 'No trained model configured'
            return
        try:
            self.backend = create_backend(self.config['backend'], self.config['model_path'],
                                          self.config['input_width'], self.config['input_height'], self.config['device'])
            self.error = ''
        except Exception as error:
            self.error = f'{type(error).__name__}: {error}'
            self.retry_after = time.monotonic() + self.config['retry_seconds']
            if not self.config['fallback_enabled']:
                raise

    def process(self, bgr, follow_right=False, branch_right=None):
        start = time.perf_counter()
        if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
            raise ValueError('Expected uint8 BGR camera frame')
        if self.backend is None and self.config['backend'] != 'opencv' and self.config['model_path'] and time.monotonic() >= self.retry_after:
            self._try_load()
        estimate, mask = None, None
        mode = 'opencv'
        if self.backend is not None:
            try:
                rgb = preprocess(bgr, self.config['input_width'], self.config['input_height'])
                logits = self.backend.logits(rgb)
                mask = decode(logits, (bgr.shape[1], bgr.shape[0]), self.config['confidence_threshold'])
                estimate = self.tracker.update(mask, follow_right, branch_right)
                mode = self.backend.name
            except Exception as error:
                self.error = f'{type(error).__name__}: {error}'
                self.close()
                self.retry_after = time.monotonic() + self.config['retry_seconds']
                if not self.config['fallback_enabled']:
                    raise
        if estimate is None or not estimate.valid:
            if self.config['backend'] == 'opencv' or self.config['fallback_enabled']:
                mask = self.classical.predict(bgr)
                estimate = self.tracker.update(mask, follow_right, branch_right)
                mode = 'opencv' if self.config['backend'] == 'opencv' else 'opencv_fallback'
        if estimate is None:
            raise RuntimeError('No segmentation backend is available')
        return estimate, mask, {'backend': mode, 'valid': estimate.valid, 'error': self.error,
                                'pipeline_ms': (time.perf_counter() - start) * 1000}

    def close(self):
        if self.backend is not None and hasattr(self.backend, 'close'):
            self.backend.close()
        self.backend = None
