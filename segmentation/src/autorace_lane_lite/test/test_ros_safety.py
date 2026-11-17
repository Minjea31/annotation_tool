"""Integration tests require a built/sourced ROS workspace (CPU/OpenCV only)."""
import time
import unittest

try:
    import cv2
    import numpy as np
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from ackermann_msgs.msg import AckermannDriveStamped
    from sensor_msgs.msg import CompressedImage
    from autorace_lane_lite.node import LaneNode
    HAVE_ROS = True
except ImportError:
    HAVE_ROS = False


@unittest.skipUnless(HAVE_ROS, 'Source ROS and the built lane workspace first')
class ROSSafetyTests(unittest.TestCase):
    def setUp(self):
        rclpy.init(args=['--ros-args', '-p', 'backend:=opencv', '-p',
                        'bev_source:=[0.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0]'])
        self.addCleanup(lambda: rclpy.shutdown() if rclpy.ok() else None)
        self.lane = LaneNode()
        self.peer = Node('lane_safety_test_peer')
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.lane)
        self.executor.add_node(self.peer)
        self.commands = []
        self.peer.create_subscription(AckermannDriveStamped, '/mission/lane', self.commands.append, 1)
        self.camera = self.peer.create_publisher(CompressedImage, '/camera/color/image_raw/compressed', 1)
        self.spin_for(0.2)

    def tearDown(self):
        self.executor.shutdown()
        self.lane.destroy_node()
        self.peer.destroy_node()
        rclpy.shutdown()

    def spin_for(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.01)

    def frame(self, age=0.0, corrupt=False):
        frame = CompressedImage()
        now_ns = self.peer.get_clock().now().nanoseconds - int(age * 1e9)
        frame.header.stamp.sec, frame.header.stamp.nanosec = divmod(now_ns, 10**9)
        frame.format = 'jpeg'
        image = np.zeros((480, 640, 3), np.uint8)
        cv2.line(image, (175, 0), (175, 479), (0, 165, 255), 8)
        frame.data = b'corrupt' if corrupt else cv2.imencode('.jpg', image)[1].tobytes()
        self.camera.publish(frame)

    def test_fresh_frame_moves_then_expiry_stops(self):
        for _ in range(4):
            self.frame()
            self.spin_for(0.06)
        self.assertTrue(any(command.drive.speed > 0 for command in self.commands))
        self.assertGreater(self.commands[-1].drive.speed, 0)
        count = len(self.commands)
        self.spin_for(0.08)
        self.assertEqual(len(self.commands), count, 'A processed image must not be republished as new perception')
        self.spin_for(0.3)
        self.assertEqual(self.commands[-1].drive.speed, 0)
        self.assertEqual(self.commands[-1].drive.steering_angle, 0)

    def test_stale_future_and_corrupt_frames_stop(self):
        for age, corrupt in ((1.0, False), (-1.0, False), (0.0, True)):
            self.frame(age, corrupt)
            self.spin_for(0.08)
            self.assertTrue(self.commands)
            self.assertEqual(self.commands[-1].drive.speed, 0)


if __name__ == '__main__':
    unittest.main()
