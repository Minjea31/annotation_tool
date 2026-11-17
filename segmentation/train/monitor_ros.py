"""Record /lane/status while the real camera, lane node and YOLO are running."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=60.0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.seconds <= 0 or args.output.exists():
        parser.error('Use a positive duration and a new output file')
    import rclpy
    from std_msgs.msg import String
    rclpy.init(args=[])
    node = rclpy.create_node('lane_performance_monitor')
    records = []

    def receive(message):
        record = json.loads(message.data)
        record['received_monotonic'] = time.monotonic()
        records.append(record)

    node.create_subscription(String, '/lane/status', receive, 1)
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if not records:
        raise RuntimeError('No lane status received; check the node and DDS configuration')
    names = ('processed_fps', 'valid_lane_fps', 'neural_lane_fps')
    report = {'seconds': args.seconds, 'status_samples': len(records), 'records': records}
    for name in names:
        values = [row[name] for row in records if name in row]
        report[name + '_mean'] = sum(values) / len(values) if values else None
        report[name + '_min'] = min(values) if values else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'records'}, indent=2))


if __name__ == '__main__':
    main()
