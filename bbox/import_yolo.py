"""Import a labeled YOLO archive into the managed BBox workspace."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import stat
import tempfile
import time
import zipfile

from PIL import Image
import yaml

from . import app as review


SPLITS = {'train': 'train', 'valid': 'test', 'val': 'test', 'test': 'eval'}


def extract_archive(archive, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError(f'Extraction directory already exists: {destination}')
    with zipfile.ZipFile(archive) as source:
        # Validate every member before writing any files.
        for member in source.infolist():
            target = (destination/member.filename).resolve()
            if not target.is_relative_to(destination) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f'Unsafe ZIP member: {member.filename}')
        destination.mkdir(parents=True)
        for member in source.infolist():
            target = destination/member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            data = source.read(member)
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError(f'Conflicting ZIP members: {member.filename}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return destination


def read_dataset(source):
    source = Path(source).resolve()
    metadata = yaml.safe_load((source/'data.yaml').read_text())
    names = metadata['names']
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    if not names or len(set(names)) != len(names) or not set(names).issubset(review.NAMES):
        raise ValueError(f'Unsupported source classes: {names}')
    if metadata.get('nc', len(names)) != len(names):
        raise ValueError('Class count does not match names')
    mapping = {i: review.NAMES.index(name) for i, name in enumerate(names)}
    records, digests, used_labels = [], set(), set()
    for folder, split in SPLITS.items():
        for path in sorted((source/folder/'images').rglob('*')):
            if not path.is_file() or path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                continue
            relative = path.relative_to(source/folder/'images')
            label = (source/folder/'labels'/relative).with_suffix('.txt')
            if not label.is_file():
                raise ValueError(f'Missing label: {label}')
            used_labels.add(label)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in digests:
                raise ValueError(f'Duplicate image; resolve before splitting: {path}')
            digests.add(digest)
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
            boxes = []
            for number, line in enumerate(label.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    class_id, x, y, w, h = map(float, line.split())
                    if not all(math.isfinite(v) for v in (class_id, x, y, w, h)):
                        raise ValueError('Non-finite coordinates')
                    if class_id != int(class_id) or int(class_id) not in mapping or min(w, h) <= 0:
                        raise ValueError('Invalid class or box size')
                    x1, y1, x2, y2 = x-w/2, y-h/2, x+w/2, y+h/2
                    if min(x1, y1) < -1e-6 or max(x2, y2) > 1+1e-6:
                        raise ValueError('Box outside image')
                    boxes.append({'id': review.uid(), 'class_id': mapping[int(class_id)],
                                  'x1': max(0, x1)*width, 'y1': max(0, y1)*height,
                                  'x2': min(1, x2)*width, 'y2': min(1, y2)*height})
                except (ValueError, OverflowError) as error:
                    raise ValueError(f'{label}:{number}: {error}') from error
            records.append({'source': str(path.relative_to(source)), 'path': path, 'name': path.name,
                            'digest': digest, 'width': width, 'height': height, 'boxes': boxes, 'split': split})
    labels = {path for folder in SPLITS for path in (source/folder/'labels').rglob('*.txt')}
    if labels != used_labels:
        raise ValueError('Labels without matching images')
    if not records or any(not any(row['split']==split for row in records) for split in ('train', 'test', 'eval')):
        raise ValueError('Expected labeled train, validation and test partitions')
    return records, names


def build_dataset(source, destination, records, names, archive):
    old_root, old_work = review.ROOT, review.WORK
    review.ROOT = Path(destination).resolve()
    review.WORK = review.ROOT/'_store/review'
    try:
        review.initialize()
        registry = {'policy': 'fixed_eval_v1', 'id': review.uid(), 'eval_seed': 31,
                    'view_seed': 31, 'entries': {}, 'hash_groups': {}}
        imported = []
        with sqlite3.connect(review.WORK/'review.sqlite') as connection:
            for record in records:
                image_id, revision_id = review.uid(), review.uid()
                image_path = review.WORK/'images'/(image_id+record['path'].suffix.lower())
                label_path = review.WORK/'labels'/(revision_id+'.json')
                shutil.copy2(record['path'], image_path)
                review.atomic(label_path, record['boxes'])
                origin = record['digest']
                connection.execute('INSERT INTO images VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                    (image_id, record['name'], str(image_path), record['digest'], origin, Path(source).name,
                     record['width'], record['height'], 'approved', revision_id, None))
                connection.execute('INSERT INTO revisions VALUES(?,?,?,?,?,?)',
                    (revision_id, image_id, 'imported', str(label_path), None, time.time()))
                group = 'eval' if record['split']=='eval' else 'pool'
                registry['entries'][origin] = {'group': group, 'sha256': record['digest'],
                                              'revision': None, 'view_split': record['split']}
                registry['hash_groups'][record['digest']] = group
                imported.append({'id': image_id, 'source': record['source'], 'sha256': record['digest'],
                                 'revision': revision_id, 'split': record['split'], 'boxes': len(record['boxes'])})
        review.atomic(review.ROOT/'_store/registry.json', registry)
        review.sync()
        snapshot = review.snapshot(seed=31)
        counts = dict(Counter(record['split'] for record in records))
        manifest = json.loads((review.ROOT/'manifest.json').read_text())
        actual = {row['id']: row['split'] for row in manifest['samples']}
        if actual != {row['id']: row['split'] for row in imported}:
            raise ValueError('Imported split membership changed')
        report = {'source': str(source), 'archive': str(archive),
                  'archive_sha256': hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
                  'source_classes': names, 'classes': review.NAMES,
                  'class_mapping': {name: review.NAMES.index(name) for name in names},
                  'images': len(records), 'boxes': sum(len(row['boxes']) for row in records),
                  'counts': counts, 'snapshot': snapshot, 'samples': imported,
                  'class_counts': {name: sum(box['class_id']==i for row in records for box in row['boxes'])
                                   for i, name in enumerate(review.NAMES)}}
        review.atomic(review.ROOT/'_store/imports'/f'{review.uid()}.json', report)
        (review.ROOT/'.gitkeep').touch()
        return report
    finally:
        review.ROOT, review.WORK = old_root, old_work


def relocate_paths(stage, destination):
    """Keep stored paths valid after atomically installing the prepared workspace."""
    prefix, replacement = str(stage.resolve())+'/', str(destination.resolve())+'/'
    def relocate(value):
        if isinstance(value, str):
            return replacement+value[len(prefix):] if value.startswith(prefix) else value
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        return value
    for path in list(stage.rglob('*.json'))+list(stage.rglob('*.yaml')):
        value = json.loads(path.read_text())  # Generated dataset.yaml is JSON-compatible YAML.
        review.atomic(path, relocate(value))
    with sqlite3.connect(stage/'_store/review/review.sqlite') as connection:
        for table in ('images', 'revisions'):
            for row_id, path in connection.execute(f'SELECT id,path FROM {table}').fetchall():
                connection.execute(f'UPDATE {table} SET path=? WHERE id=?', (relocate(path), row_id))
    return relocate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--replace', action='store_true', help='Back up and replace the current BBox dataset')
    args = parser.parse_args()
    destination = review.ROOT
    if destination.exists() and not args.replace:
        parser.error('Use --replace to back up and replace the BBox dataset')
    source = extract_archive(args.archive.resolve(), args.archive.resolve().with_suffix(''))
    records, names = read_dataset(source)
    backups = review.HERE.parent/'backups'
    backups.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.bbox-import-', dir=backups) as directory:
        stage = Path(directory)/'dataset'
        report = build_dataset(source, stage, records, names, args.archive.resolve())
        relocate = relocate_paths(stage, destination)
        database = destination/'_store/review/review.sqlite'
        if database.exists():
            with sqlite3.connect(database) as connection:
                if connection.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running') LIMIT 1").fetchone():
                    raise RuntimeError('Wait for BBox inference/training before replacing its dataset')
        backup = backups/('bbox-dataset-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        if destination.exists():
            destination.rename(backup)
        try:
            stage.rename(destination)
        except BaseException:
            if backup.exists():
                backup.rename(destination)
            raise
        summary = relocate({key: value for key, value in report.items() if key!='samples'})
        summary['backup'] = str(backup) if backup.exists() else None
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
