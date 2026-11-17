"""Shared semantic-mask → BEV → boundary tracking → Ackermann steering."""
from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class Estimate:
    valid: bool = False
    speed: float = 0.0
    steering: float = 0.0
    center_offset_m: float = 0.0
    heading_rad: float = 0.0
    boundaries: int = 0
    branch: bool = False


class LaneTracker:
    def __init__(self, config):
        self.config = config
        self.width, self.height = config['bev_width'], config['bev_height']
        source = np.asarray(config['bev_source'], np.float32).reshape(4, 2)
        if not np.isfinite(source).all() or (source < 0).any() or (source > 1).any():
            raise ValueError('BEV source coordinates must be normalized to [0,1]')
        if abs(cv2.contourArea(source)) < 0.001 or not cv2.isContourConvex(source):
            raise ValueError('BEV source must form a convex quadrilateral: TL,TR,BR,BL')
        self.source = source
        self.transform, self.image_size = None, None
        self.previous = None
        self.bev = np.zeros((self.height, self.width), np.uint8)

    def reset(self):
        self.previous = None

    def warp(self, mask):
        height, width = mask.shape
        if self.image_size != (width, height):
            source = self.source * np.array([width - 1, height - 1], np.float32)
            target = np.array([[0, 0], [self.width - 1, 0],
                               [self.width - 1, self.height - 1], [0, self.height - 1]], np.float32)
            self.transform = cv2.getPerspectiveTransform(source, target)
            self.image_size = width, height
        self.bev = cv2.warpPerspective(mask, self.transform, (self.width, self.height), flags=cv2.INTER_NEAREST)
        return self.bev

    def _fit(self, binary, base):
        ys, xs = np.nonzero(binary)
        margin = max(4, int(self.width * self.config['window_margin_ratio']))
        center, selected, covered = float(base), [], 0
        for bottom in np.linspace(self.height, 0, 10, dtype=int)[:-1]:
            top = max(0, bottom - math.ceil(self.height / 9))
            keep = (ys >= top) & (ys < bottom) & (np.abs(xs - center) < margin)
            if keep.sum() >= self.config['window_min_pixels']:
                center = float(np.median(xs[keep]))
                selected.append(np.flatnonzero(keep))
                covered += 1
        if covered < self.config['min_windows']:
            return None
        indices = np.unique(np.concatenate(selected))
        forward = (self.height - 1 - ys[indices]) / (self.height - 1)
        if np.ptp(forward) < 0.25:
            return None
        xx = xs[indices] / (self.width - 1)
        poly = np.polyfit(forward, xx, 2)
        residual = np.abs(np.polyval(poly, forward) - xx)
        good = residual < max(0.015, float(np.median(residual)) * 3)
        if good.sum() >= 12 and np.ptp(forward[good]) >= 0.25:
            poly = np.polyfit(forward[good], xx[good], 2)
        probes = np.polyval(poly, [0.0, 0.25, 0.5, 0.75])
        return poly if np.isfinite(poly).all() and (probes > -0.2).all() and (probes < 1.2).all() else None

    def update(self, mask, follow_right=False, branch_right=None):
        binary = self.warp(mask) > 0
        histogram = binary[self.height // 2:].sum(axis=0).astype(np.float32)
        histogram = cv2.GaussianBlur(histogram[None], (9, 1), 0)[0]
        radius = max(5, int(self.width * self.config['window_margin_ratio']))
        maxima = cv2.dilate(histogram[None], np.ones((1, 2 * radius + 1), np.uint8))[0]
        candidates = np.flatnonzero((histogram >= maxima) & (histogram > max(2, histogram.max() * 0.2)))
        peaks = []
        for group in np.split(candidates, np.where(np.diff(candidates) > 1)[0] + 1):
            if len(group):
                peaks.append(float(np.mean(group)))
        fits = [poly for peak in peaks if (poly := self._fit(binary, peak)) is not None]
        fits.sort(key=lambda poly: float(np.polyval(poly, 0.0)))
        if not fits:
            self.reset()
            return Estimate()
        lane_width = self.config['lane_width_ratio']
        pairs = []
        for left, right in zip(fits, fits[1:]):
            widths = np.polyval(right - left, [0.0, 0.25, 0.5])
            if (widths > lane_width * 0.5).all() and (widths < lane_width * 1.7).all():
                pairs.append((left, right))
        branch = len(pairs) >= 2
        left, right = None, None
        if pairs:
            if branch and branch_right is not None:
                left, right = pairs[-1 if branch_right else 0]
            else:
                left, right = min(pairs, key=lambda pair: abs(np.polyval((pair[0] + pair[1]) / 2, 0) - 0.5))
        else:
            preferred = [poly for poly in fits if (np.polyval(poly, 0) > 0.5) == follow_right]
            poly = min(preferred or fits, key=lambda p: abs(abs(np.polyval(p, 0) - 0.5) - lane_width / 2))
            if np.polyval(poly, 0) < 0.5:
                left = poly
            else:
                right = poly
        if left is not None and right is not None:
            center = (left + right) / 2
        else:
            center = (left if left is not None else right).copy()
            center[-1] += lane_width / 2 if left is not None else -lane_width / 2
        if self.previous is not None and abs(np.polyval(center - self.previous, 0)) < 0.15:
            alpha = self.config['tracking_alpha']
            center = alpha * center + (1 - alpha) * self.previous
        self.previous = center
        lookahead = self.config['lookahead_m']
        normalized_y = lookahead / self.config['bev_forward_m']
        lateral = (0.5 - np.polyval(center, normalized_y)) * self.config['bev_lateral_m']
        slope = -np.polyval(np.polyder(center), normalized_y) * self.config['bev_lateral_m'] / self.config['bev_forward_m']
        heading = math.atan(slope)
        steering = math.atan2(2 * self.config['wheelbase_m'] * lateral, lookahead ** 2 + lateral ** 2)
        steering += self.config['heading_gain'] * heading
        steering = float(np.clip(steering, -self.config['max_steering_rad'], self.config['max_steering_rad']))
        speed = self.config['speed_mps'] / (1 + self.config['curve_slowdown'] * abs(steering))
        if abs(lateral) > self.config['max_center_offset_m']:
            self.reset()
            return Estimate()
        return Estimate(True, float(speed), steering, float(lateral), heading,
                        int(left is not None) + int(right is not None), branch)
