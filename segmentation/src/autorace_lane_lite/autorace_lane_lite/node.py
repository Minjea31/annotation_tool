import json
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from ament_index_python.packages import get_package_share_directory
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, String

from .configuration import image_is_fresh, load_config, validate_config
from .pipeline import LanePipeline


class LaneNode(Node):
    def __init__(self):
        super().__init__('lane_node')
        defaults = load_config(Path(get_package_share_directory('autorace_lane_lite')) / 'config' / 'lane.yaml')
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.config = {name: self.get_parameter(name).value for name in defaults}
        validate_config(self.config)
        self.pipeline = LanePipeline(self.config)
        self.add_on_set_parameters_callback(self._validate_parameters)
        self.latest = None
        self.processed = None
        self.follow_right, self.sign = False, None
        self.last_report, self.frames = time.monotonic(), 0
        self.valid_frames, self.neural_frames = 0, 0
        self.last_mode = None
        command_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.VOLATILE)
        camera_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.command_pub = self.create_publisher(AckermannDriveStamped, '/mission/lane', command_qos)
        self.status_pub = self.create_publisher(String, '/lane/status', 1)
        self.mask_pub = self.create_publisher(Image, '/lane/mask', 1) if self.config['debug_masks'] else None
        self.bev_pub = self.create_publisher(Image, '/lane/bev_mask', 1) if self.config['debug_masks'] else None
        self.create_subscription(CompressedImage, self.config['camera_topic'], self._image, camera_qos)
        self.create_subscription(Bool, '/current_lane', self._lane, command_qos)
        self.create_subscription(Bool, '/yolo/LR_sign', self._sign, command_qos)
        self.timer = self.create_timer(1.0 / self.config['process_hz'], self._tick)
        self.get_logger().info(f"Lane backend={self.config['backend']}, model={self.config['model_path'] or '(none)'}, fallback={self.config['fallback_enabled']}")
        if self.pipeline.error:
            self.get_logger().warning(self.pipeline.error)

    def _validate_parameters(self, parameters):
        # The model, geometry and timer are constructed together; avoid partial live updates.
        fixed = [p.name for p in parameters if p.name in self.config and p.value != self.config[p.name]]
        return SetParametersResult(successful=not fixed,
                                   reason='Restart lane_node to change configuration' if fixed else '')

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _image(self, message):
        self.latest = (message, time.monotonic())

    def _lane(self, message):
        self.follow_right = message.data

    def _sign(self, message):
        self.sign = (message.data, self._now(), time.monotonic())

    def _publish(self, speed=0.0, steering=0.0):
        command = AckermannDriveStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = 'base_link'
        command.drive.speed, command.drive.steering_angle = float(speed), float(steering)
        self.command_pub.publish(command)

    def _fresh(self, message, receipt):
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        return image_is_fresh(stamp, self._now(), receipt, time.monotonic(), self.config)

    def _status(self, data):
        now = time.monotonic()
        self.frames += int(data.get('processed', False))
        self.valid_frames += int(data.get('valid', False))
        self.neural_frames += int(data.get('valid', False) and data['backend'] in ('torch', 'onnx', 'tensorrt'))
        if now - self.last_report >= 1.0:
            data['processed_fps'] = self.frames / (now - self.last_report)
            data['valid_lane_fps'] = self.valid_frames / (now - self.last_report)
            data['neural_lane_fps'] = self.neural_frames / (now - self.last_report)
            self.status_pub.publish(String(data=json.dumps(data)))
            self.last_report, self.frames = now, 0
            self.valid_frames, self.neural_frames = 0, 0
        mode = (data['backend'], data['valid'], data.get('error', ''))
        if mode != self.last_mode:
            self.get_logger().info(f'Lane mode={mode[0]}, valid={mode[1]}, detail={mode[2]}')
            self.last_mode = mode

    def _debug(self, publisher, mask, header):
        if publisher is None or publisher.get_subscription_count() == 0:
            return
        message = Image()
        message.header = header
        message.height, message.width = mask.shape
        message.encoding, message.step = 'mono8', mask.shape[1]
        message.data = mask.tobytes()
        publisher.publish(message)

    def _tick(self):
        if self.latest is None:
            self._publish()
            self._status({'backend': 'waiting_camera', 'valid': False})
            return
        message, receipt = self.latest
        if not self._fresh(message, receipt):
            self.pipeline.tracker.reset()
            self._publish()
            self._status({'backend': 'stale_camera', 'valid': False})
            return
        if message is self.processed:
            return  # Do not give old perception a fresh command timestamp.
        self.processed = message
        start = time.perf_counter()
        try:
            image = cv2.imdecode(np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('Camera JPEG/PNG decoding failed')
            direction = None
            if self.sign is not None:
                value, received_ros, received_wall = self.sign
                ttl = self.config['sign_ttl_seconds']
                if 0 <= self._now() - received_ros <= ttl and 0 <= time.monotonic() - received_wall <= ttl:
                    direction = value
            estimate, mask, stats = self.pipeline.process(image, self.follow_right, direction)
            stats['processed'] = True
            if not self._fresh(message, receipt):
                self.pipeline.tracker.reset()
                self._publish()
                stats.update(valid=False, error='Image expired during inference')
            else:
                self._publish(estimate.speed, estimate.steering)
                self._debug(self.mask_pub, mask, message.header)
                self._debug(self.bev_pub, self.pipeline.tracker.bev, message.header)
            stats['total_ms'] = (time.perf_counter() - start) * 1000
            stats['center_offset_m'], stats['heading_rad'] = estimate.center_offset_m, estimate.heading_rad
            self._status(stats)
        except Exception as error:
            self.pipeline.tracker.reset()
            self._publish()
            self._status({'backend': 'error', 'valid': False, 'error': str(error)})

    def destroy_node(self):
        if hasattr(self, 'pipeline'):
            self.pipeline.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LaneNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
