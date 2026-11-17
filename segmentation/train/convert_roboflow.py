"""Convert Roboflow COCO instance polygons into indexed semantic masks."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401
from autorace_lane_lite.constants import CLASSES


def polygon_mask(annotation, height, width):
    polygons = annotation.get('segmentation')
    if not isinstance(polygons, list) or not polygons:
        raise ValueError('Expected COCO polygon segmentation; keypoints/bboxes/RLE are not accepted')
    mask = np.zeros((height, width), np.uint8)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float64)
        if points.size < 6 or points.size % 2 or not np.isfinite(points).all():
            raise ValueError('Invalid polygon coordinates')
        points = points.reshape(-1, 2)
        if abs(cv2.contourArea(points.astype(np.float32))) < 0.5:
            raise ValueError('Degenerate polygon')
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask.astype(bool)


def convert(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'Output already exists: {destination}; choose a new directory')
    files = sorted(source.rglob('_annotations.coco.json'))
    if not files:
        raise ValueError('No _annotations.coco.json found; export COCO instance segmentation')
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest, hashes, counts = [], {}, {}
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.lane-convert-') as staging:
        root = Path(staging) / 'dataset'
        root.mkdir()
        for annotation_file in files:
            split = {'valid': 'val', 'val': 'val', 'train': 'train', 'test': 'test'}.get(annotation_file.parent.name)
            if split is None:
                raise ValueError(f'Expected train/valid/test directory: {annotation_file}')
            data = json.loads(annotation_file.read_text())
            categories = {c['id']: c['name'] for c in data['categories']}
            images = {im['id']: im for im in data['images']}
            if len(images) != len(data['images']):
                raise ValueError('Duplicate COCO image IDs')
            grouped = {key: [] for key in images}
            for ann in data['annotations']:
                if ann['image_id'] not in images:
                    raise ValueError('Annotation refers to an unknown image')
                name = categories.get(ann['category_id'])
                if name not in CLASSES[1:]:
                    raise ValueError(f'Unexpected annotated class {name!r}; expected {CLASSES[1:]}')
                grouped[ann['image_id']].append(ann)
            for image_id, info in images.items():
                path = (annotation_file.parent / info['file_name']).resolve()
                if not path.is_relative_to(annotation_file.parent.resolve()):
                    raise ValueError('COCO image path escapes its split directory')
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in hashes:
                    raise ValueError(f'Duplicate image bytes in {hashes[digest]} and {split}: {path.name}')
                hashes[digest] = split
                with Image.open(path) as image:
                    image = image.convert('RGB')
                    width, height = image.size
                    if (width, height) != (info['width'], info['height']):
                        raise ValueError(f'Image dimensions disagree with COCO: {path}')
                    planes = np.zeros((2, height, width), dtype=bool)
                    for ann in grouped[image_id]:
                        index = CLASSES.index(categories[ann['category_id']]) - 1
                        planes[index] |= polygon_mask(ann, height, width)
                    mask = np.zeros((height, width), np.uint8)
                    mask[planes[0]], mask[planes[1]] = 1, 2
                    mask[planes.all(axis=0)] = 255  # Ambiguous overlap: ignore in loss/metrics.
                    stem = f'{split}_{digest[:20]}'
                    for directory in ('images', 'masks'):
                        (root / split / directory).mkdir(parents=True, exist_ok=True)
                    image_rel, mask_rel = f'{split}/images/{stem}.png', f'{split}/masks/{stem}.png'
                    image.save(root / image_rel)
                    Image.fromarray(mask).save(root / mask_rel)
                manifest.append({'image': image_rel, 'mask': mask_rel, 'split': split,
                                 'source': str(path.relative_to(source)), 'sha256': digest})
                counts[split] = counts.get(split, np.zeros(4, dtype=np.int64)) + np.array(
                    [(mask == c).sum() for c in (0, 1, 2, 255)])
        if not any(row['split'] == 'train' for row in manifest):
            raise ValueError('Training split is missing')
        metadata = {'classes': list(CLASSES), 'ignore_index': 255, 'samples': manifest,
                    'pixel_counts_0_1_2_255': {k: v.tolist() for k, v in counts.items()}}
        (root / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
        shutil.move(str(root), str(destination))
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = convert(args.source, args.output)
    print(json.dumps({'images': len(result['samples']), 'pixel_counts': result['pixel_counts_0_1_2_255']}, indent=2))
