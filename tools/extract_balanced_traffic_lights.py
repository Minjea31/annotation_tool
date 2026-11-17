"""Select equal numbers of red/green frames from an unmodified ROS2 SQLite bag.

Run with the annotation yolo Python in a ROS2-sourced environment.
Images stay unlabeled; model detections and source timestamps go in manifest.json.
"""
import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
import sys

import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bbox.engine import load


def decode(data):
    msg = deserialize_message(data, Image)
    if msg.encoding not in ('rgb8', 'bgr8'):
        raise ValueError(f'Unsupported image encoding: {msg.encoding}')
    pixels = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    pixels = pixels[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR) if msg.encoding == 'rgb8' else pixels.copy()


def color_evidence(image, detection):
    x1, y1, x2, y2 = detection['xyxy']
    roi = image[max(0, int(y1)):min(image.shape[0], int(np.ceil(y2))),
                max(0, int(x1)):min(image.shape[1], int(np.ceil(x2)))]
    if not roi.size:
        return {'red_pixels':0, 'green_pixels':0, 'matches_prediction':False}
    h, s, v = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
    bright = (s >= 130) & (v >= 200)
    red = int(np.count_nonzero(bright & ((h <= 10) | (h >= 170))))
    green = int(np.count_nonzero(bright & (h >= 40) & (h <= 90)))
    own, other = (red, green) if detection['class'] == 'red' else (green, red)
    return {'red_pixels':red, 'green_pixels':green,
            'matches_prediction':own >= 3 and own >= 1.5 * other}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('weights', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--per-class', type=int, default=300)
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error('--per-class must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    dbs = sorted(args.bag.glob('*.db3'))
    if len(dbs) != 1:
        raise ValueError('Expected a single SQLite bag file')
    db = dbs[0]
    connection = sqlite3.connect(f'{db.resolve().as_uri()}?mode=ro', uri=True)
    topics = connection.execute("SELECT id,name FROM topics WHERE type='sensor_msgs/msg/Image'").fetchall()
    if len(topics) != 1:
        raise ValueError(f'Expected one raw image topic, found {topics}')
    topic_id, topic = topics[0]
    total = connection.execute('SELECT count(*) FROM messages WHERE topic_id=?', (topic_id,)).fetchone()[0]
    model = load(args.weights)
    cache = args.output / 'scan.json'
    signature = {'bag':str(db.resolve()), 'bag_size':db.stat().st_size,
                 'bag_mtime_ns':db.stat().st_mtime_ns,
                 'weights_sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                 'confidence':args.confidence, 'topic':topic}
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved['signature'] != signature:
            raise ValueError('Output scan belongs to different input/settings')
        frames = saved['frames']
    else:
        frames = []
        batch, metadata = [], []
        def flush():
            if not batch:
                return
            results = model.predict(batch, imgsz=640, conf=args.confidence, device=0, verbose=False)
            for row, result in zip(metadata, results):
                detections = [{'class':model.names[int(cls)], 'class_id':int(cls),
                               'confidence':float(conf), 'xyxy':[float(v) for v in xy]}
                              for xy, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())
                              if model.names[int(cls)] in ('red', 'green')]
                frames.append({**row, 'detections':detections})
            print(f'Scanned {len(frames)}/{total}', flush=True)
            batch.clear()
            metadata.clear()
        for index, (message_id, timestamp, data) in enumerate(connection.execute(
                'SELECT id,timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id', (topic_id,))):
            batch.append(decode(data))
            metadata.append({'frame_index':index, 'message_id':message_id, 'timestamp_ns':timestamp})
            if len(batch) == args.batch:
                flush()
        flush()
        cache.write_text(json.dumps({'signature':signature,'frames':frames}, indent=2))
    counts = Counter()
    for frame in frames:
        present = {d['class'] for d in frame['detections']}
        counts['mixed' if len(present) > 1 else next(iter(present), 'none')] += 1
    print('Candidate counts:', dict(counts), flush=True)
    selected = []
    for name in ('red', 'green'):
        candidates = [f for f in frames if {d['class'] for d in f['detections']} == {name}]
        for row in candidates:
            if 'color_evidence' not in row:
                raw = connection.execute('SELECT data FROM messages WHERE id=?', (row['message_id'],)).fetchone()[0]
                detection = max(row['detections'], key=lambda d:d['confidence'])
                row['color_evidence'] = color_evidence(decode(raw), detection)
        candidates = [f for f in candidates if f['color_evidence']['matches_prediction']]
        if len(candidates) < args.per_class:
            raise ValueError(f'Only {len(candidates)} exclusive {name} frames; need {args.per_class}')
        # Use higher-confidence candidates when sufficient, then spread over their timeline.
        strong = [f for f in candidates if max(d['confidence'] for d in f['detections']) >= .5]
        if len(strong) >= args.per_class:
            candidates = strong
        indices = np.linspace(0, len(candidates)-1, args.per_class).round().astype(int)
        folder = args.output / 'images' / name
        folder.mkdir(parents=True, exist_ok=True)
        for index in indices:
            row = candidates[int(index)]
            filename = f"{args.bag.name}__frame_{row['frame_index']:06d}_{name}.png"
            relative = Path('images') / name / filename
            raw = connection.execute('SELECT data FROM messages WHERE id=?', (row['message_id'],)).fetchone()[0]
            image = decode(raw)
            if not cv2.imwrite(str(args.output / relative), image):
                raise IOError(f'Cannot write {relative}')
            selected.append({**row, 'selected_class':name, 'file':str(relative),
                             'width':image.shape[1], 'height':image.shape[0],
                             'sha256':hashlib.sha256((args.output / relative).read_bytes()).hexdigest()})
    # Remove only obsolete images recorded by this extractor's previous manifest.
    previous = args.output / 'manifest.json'
    keep = {row['file'] for row in selected}
    if previous.exists():
        for row in json.loads(previous.read_text()):
            obsolete = (args.output / row['file']).resolve()
            if row['file'] not in keep and obsolete.is_relative_to((args.output/'images').resolve()):
                obsolete.unlink(missing_ok=True)
    cache.write_text(json.dumps({'signature':signature,'frames':frames}, indent=2))
    files = list((args.output / 'images').rglob('*.png'))
    if len(files) != args.per_class * 2:
        raise ValueError(f'Unexpected image count {len(files)}')
    if len({row['sha256'] for row in selected}) != len(selected):
        raise ValueError('Selected images contain identical pixels; revise selection')
    summary = {**signature, 'total_bag_frames':total, 'candidate_counts':dict(counts),
               'selected_counts':dict(Counter(r['selected_class'] for r in selected)),
               'image_count':len(selected), 'import_path':str((args.output/'images').resolve()),
               'selection':'Exclusive model class; bright HSV pixels confirm predicted color; prefer confidence >=0.5; evenly spaced candidate indices',
               'labels':'Unlabeled images; detections in manifest are model predictions for human review'}
    (args.output/'manifest.json').write_text(json.dumps(selected, indent=2))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
