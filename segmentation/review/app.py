"""Lane review server with shared local storage. Run with the training Python environment."""
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import re
import random
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
import dataset_layout
DATA_ROOT = Path(os.environ.get('ANNOTATION_SEG_DATA', HERE.parent/'dataset')).resolve()
TRAIN = HERE.parent / 'train'
sys.path.insert(0, str(TRAIN))
import _bootstrap  # noqa
import dataset_store
WORK = Path(os.environ.get('LANE_REVIEW_WORK', DATA_ROOT / '_store/review')).resolve()
APPROVED = Path(os.environ.get('LANE_REVIEW_APPROVE', DATA_ROOT / '_store/pending')).resolve()
DELETED = Path(os.environ.get('LANE_REVIEW_DELETE', DATA_ROOT / 'deleted')).resolve()
BASE = DATA_ROOT / '_store/baseline_data/autorace_deep_v1_20261106'
INITIAL = HERE.parent / 'models/best.pt'
POOL = ThreadPoolExecutor(max_workers=1)
LOCK = threading.RLock()


def uid():
    return uuid.uuid4().hex


def db():
    connection = sqlite3.connect(WORK / 'review.sqlite', timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def query(sql, values=(), one=False):
    with db() as c:
        rows = c.execute(sql, values).fetchall()
    return (dict(rows[0]) if rows else None) if one else [dict(r) for r in rows]


def update(sql, values=()):
    with db() as c:
        c.execute(sql, values)


def item(image_id):
    row = query('SELECT * FROM images WHERE id=?', (image_id,), True)
    if not row:
        raise HTTPException(404, 'Image not found')
    return row


def identity(name, digest, fallback):
    m = re.match(r'(.*?)__frame_(\d+)', Path(name).name)
    session = m[1] if m else fallback
    return session, f'{session}:{int(m[2])}' if m else digest


def revision(image_id, kind, path, model=None):
    revision_id = uid()
    update('INSERT INTO revisions VALUES (?,?,?,?,?,?)',
           (revision_id, image_id, kind, str(path), model, time.time()))
    return revision_id


def add_image(path, name, session_hint, mask=None, split=None):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    session, origin = identity(name, digest, session_hint)
    if query('SELECT id FROM images WHERE digest=? OR origin=?', (digest, origin), True):
        return False
    with Image.open(path) as im:
        im.load()
        width, height = im.size
        if width * height > 25000000:
            raise ValueError('Image too large')
    split = 'train'  # Pool membership is independent of the per-run 6:2:2 split.
    image_id = uid()
    image_path = WORK / 'images' / f'{image_id}{path.suffix.lower()}'
    shutil.copy2(path, image_path)
    current = None
    if mask:
        target = WORK / 'masks' / f'{uid()}.png'
        shutil.copy2(mask, target)
        current = revision(image_id, 'imported', target)
    update('INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
           (image_id, name, str(image_path), digest, origin, session, split,
            width, height, 'approved' if mask else 'unreviewed', current, None, time.time()))
    return True


def dataset_root():
    return Path(os.environ.get('LANE_REVIEW_DATASET', DATA_ROOT / '_store')).resolve()


def eligible_rows():
    return query("""SELECT images.*, revisions.kind AS revision_kind, revisions.path AS mask_path
                    FROM images JOIN revisions ON images.current=revisions.id
                    WHERE images.status='approved'
                       OR (images.status='editing' AND revisions.kind='edited')
                    ORDER BY images.origin""")


def backup_policy_migration():
    root = dataset_root()
    if (root / 'migration_backup').exists():
        return
    destination = root / 'migration_backup'
    destination.mkdir(parents=True)
    with db() as source, sqlite3.connect(destination / 'review.sqlite') as backup:
        source.backup(backup)
    if (APPROVED / 'manifest.json').exists():
        shutil.copy2(APPROVED / 'manifest.json', destination / 'approve_manifest.json')
    (destination / 'README.txt').write_text('Before fixed eval migration. Existing model files and version directories are retained in their original locations.\n')


def merge_pending():
    with LOCK:
        pool = sync_approved()
        pending = [{**r, '_image': str(APPROVED / r['image']), '_mask': str(APPROVED / r['mask'])} for r in pool['samples']]
        registry = dataset_store.merge(dataset_root(), pending, atomic_copy)
        sync_approved()  # Consume only revisions safely committed to the registry.
        return registry


def initialize():
    for name in ['images', 'masks', 'datasets', 'runs', 'logs']:
        (WORK / name).mkdir(parents=True, exist_ok=True)
    with db() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS images (id TEXT PRIMARY KEY, name TEXT, path TEXT,
          digest TEXT UNIQUE, origin TEXT UNIQUE, session TEXT, split TEXT, width INTEGER,
          height INTEGER, status TEXT, current TEXT, prediction TEXT, updated REAL);
        CREATE TABLE IF NOT EXISTS revisions (id TEXT PRIMARY KEY, image_id TEXT, kind TEXT,
          path TEXT, model TEXT, created REAL);
        CREATE TABLE IF NOT EXISTS models (id TEXT PRIMARY KEY, name TEXT, path TEXT UNIQUE);
        CREATE TABLE IF NOT EXISTS model_policies (model_id TEXT PRIMARY KEY, policy_id TEXT);
        CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, kind TEXT, state TEXT, payload TEXT,
          result TEXT, error TEXT, created REAL);
        ''')
        c.execute("UPDATE jobs SET state='interrupted',error='Server restarted; partial output retained' WHERE state IN ('queued','running')")
    if BASE.exists():
        manifest = json.loads((BASE / 'manifest.json').read_text())
        for row in manifest['samples']:
            add_image(BASE / row['image'], Path(row['source']).name,
                      row.get('capture_session', 'baseline'), BASE / row['mask'], row['split'])
    backup_policy_migration()
    sync_approved()
    if dataset_store.read(dataset_root()) is None and len(eligible_rows()) >= 5:
        merge_pending()
    if INITIAL.exists():
        register_model(INITIAL, '최신 모델 · MobileNetV3 U-Net · IoU 95.6%')


def register_model(path, name):
    path = path.resolve()
    if not path.is_file() or path.suffix != '.pt':
        raise HTTPException(400, 'Existing .pt path required')
    existing = query('SELECT * FROM models WHERE path=?', (str(path),), True)
    if existing:
        return existing
    model_id = uid()
    update('INSERT INTO models VALUES (?,?,?)', (model_id, name, str(path)))
    return query('SELECT * FROM models WHERE id=?', (model_id,), True)


@asynccontextmanager
async def lifespan(app):
    initialize()
    yield


app = FastAPI(lifespan=lifespan)


@app.middleware('http')
async def local_origin(request: Request, call_next):
    origin = request.headers.get('origin')
    if origin and origin.rstrip('/') != str(request.base_url).rstrip('/'):
        return __import__('fastapi').responses.JSONResponse({'detail': 'Local same-origin access only'}, status_code=403)
    return await call_next(request)


class ImportRequest(BaseModel):
    path: str
    session: str = ''


class SaveRequest(BaseModel):
    mask: str
    expected_revision: str | None = None
    approve: bool = False


class StatusRequest(BaseModel):
    status: str
    expected_revision: str | None = None


class ApplyRequest(BaseModel):
    revision: str


class InferRequest(BaseModel):
    model: str
    images: list[str] = Field(min_length=1)


class ModelRequest(BaseModel):
    path: str
    name: str = ''


class TrainRequest(BaseModel):
    model: str = 'fresh'
    epochs: int = Field(default=80, ge=1, le=1000)
    batch_size: int = Field(default=4, ge=1, le=32)
    lr: float = Field(default=0.001, gt=0, le=0.1)
    seed: int = 31


IMAGE_FIELDS = ('id', 'name', 'session', 'split', 'width', 'height', 'status', 'current', 'prediction', 'updated')


def client_image(row):
    return {key: row[key] for key in IMAGE_FIELDS}


@app.get('/api/state')
def state(compact: bool = False, image_id: str | None = None):
    fields = ','.join(IMAGE_FIELDS) if compact else '*'
    images = query(f'SELECT {fields} FROM images ORDER BY session,name')
    jobs = query('SELECT * FROM jobs ORDER BY created DESC LIMIT 30')
    for job in jobs:
        job['payload'] = json.loads(job['payload'])
        job['result'] = json.loads(job['result']) if job['result'] else None
        if compact and 'images' in job['payload']:
            selected = job['payload'].pop('images')
            job['payload']['image_count'] = len(selected)
            job['payload']['current_image'] = image_id if image_id in selected else None
        metric_file = WORK / 'runs' / job['id'] / 'metrics.jsonl'
        job['metrics'] = []
        if metric_file.exists():
            for line in metric_file.read_text().splitlines():
                try:
                    job['metrics'].append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    registry = dataset_store.read(dataset_root())
    policy_id = registry['id'] if registry else None
    models = query('SELECT * FROM models')
    for model in models:
        record = query('SELECT * FROM model_policies WHERE model_id=?', (model['id'],), True)
        model['train_eligible'] = bool(record and record['policy_id']==policy_id)
        model['legacy'] = not model['train_eligible']
    eligible = eligible_rows()
    entries = registry['entries'] if registry else {}
    counts = {'pool': 0, 'eval': 0, 'pending': 0}
    for row in eligible:
        entry = entries.get(row['origin'])
        if not entry or entry['revision']!=row['current']:
            counts['pending'] += 1
        if entry:
            counts[entry['group']] += 1
    return {'images': images, 'models': models, 'jobs': jobs,
            'dataset': {'path': str(dataset_root()), 'policy_id': policy_id, **counts},
            'work': str(WORK), 'approve_folder': str(APPROVED), 'delete_folder': str(DELETED), 'default_folder': str(DATA_ROOT / 'unlabeled')}


@app.post('/api/import')
def import_images(body: ImportRequest):
    source = Path(body.path).expanduser().resolve()
    if not source.is_dir() or source == WORK or WORK in source.parents:
        raise HTTPException(400, 'Select an external image folder')
    count, skipped, errors = 0, 0, []
    with LOCK:
        previous_ids = {r['id'] for r in query('SELECT id FROM images')}
        for path in sorted(source.rglob('*')):
            if path.resolve().is_relative_to(WORK) or path.resolve().is_relative_to(APPROVED) or path.resolve().is_relative_to(DELETED):
                continue
            if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                continue
            try:
                if add_image(path, path.name, body.session.strip() or source.name):
                    count += 1
                else:
                    skipped += 1
            except Exception as e:
                errors.append(f'{path.name}: {e}')
        imported=[r for r in query('SELECT * FROM images') if r['id'] not in previous_ids]
    dataset_layout.archive_imports(source, WORK, DATA_ROOT, query('SELECT * FROM images'))
    publish_layout()
    dataset=dataset_layout.record_import(DATA_ROOT,source,imported,skipped,errors)
    return {'added': count, 'duplicates': skipped, 'errors': errors, 'scanned': count+skipped+len(errors), 'dataset':dataset}


@app.get('/api/images/{image_id}/file')
def image_file(image_id: str):
    return FileResponse(item(image_id)['path'])


@app.get('/api/images/{image_id}/revisions')
def revisions(image_id: str):
    item(image_id)
    return query('SELECT * FROM revisions WHERE image_id=? ORDER BY created DESC', (image_id,))


@app.get('/api/revisions/{revision_id}/mask')
def mask_file(revision_id: str):
    row = query('SELECT * FROM revisions WHERE id=?', (revision_id,), True)
    if not row:
        raise HTTPException(404, 'Revision not found')
    return FileResponse(row['path'], media_type='image/png')


def editable(row):
    if row['status'] == 'deleted':
        raise HTTPException(409, 'Image archived in delete_dataset; editing is disabled')
    return row


@app.post('/api/images/{image_id}/mask')
def save_mask(image_id: str, body: SaveRequest):
    with LOCK:
        row = item(image_id)
        editable(row)
        if body.expected_revision != row['current']:
            raise HTTPException(409, 'Label changed; reload before saving')
        try:
            raw = base64.b64decode(body.mask.split(',')[-1], validate=True)
            with Image.open(io.BytesIO(raw)) as im:
                if im.size != (row['width'], row['height']):
                    raise ValueError('Mask dimensions must match image')
                pixels = np.array(im)
                if pixels.ndim == 3:
                    if not np.array_equal(pixels[..., 0], pixels[..., 1]) or not np.array_equal(pixels[..., 0], pixels[..., 2]):
                        raise ValueError('Expected class ID mask')
                    pixels = pixels[..., 0]
                if not np.isin(pixels, [0, 1, 2, 255]).all():
                    raise ValueError('Invalid class IDs')
        except Exception as e:
            raise HTTPException(400, str(e))
        path = WORK / 'masks' / f'{uid()}.png'
        Image.fromarray(pixels.astype(np.uint8)).save(path)
        rev = revision(image_id, 'edited', path)
        update('UPDATE images SET current=?,status=?,updated=? WHERE id=?',
               (rev, 'approved' if body.approve else 'editing', time.time(), image_id))
        sync_approved()
        return {'revision': rev, 'image': client_image(item(image_id))}


@app.post('/api/images/{image_id}/status')
def set_status(image_id: str, body: StatusRequest):
    with LOCK:
        row = item(image_id)
        editable(row)
        if 'expected_revision' in body.model_fields_set and body.expected_revision != row['current']:
            raise HTTPException(409, '라벨이 변경되었습니다. 최신 라벨을 확인한 뒤 다시 승인하세요.')
        if body.status not in ['approved', 'editing', 'excluded', 'retry', 'unreviewed']:
            raise HTTPException(400, 'Unknown status')
        if body.status == 'approved' and not row['current']:
            raise HTTPException(400, 'Save a mask first; empty masks may also be approved')
        update('UPDATE images SET status=?,updated=? WHERE id=?', (body.status, time.time(), image_id))
        sync_approved()
        return {'ok': True, 'image': client_image(item(image_id))}


@app.post('/api/images/{image_id}/delete')
def delete_image(image_id: str):
    with LOCK:
        row = item(image_id)
        DELETED.mkdir(parents=True, exist_ok=True)
        image = Path('images') / (image_id + Path(row['path']).suffix)
        atomic_copy(row['path'], DELETED / image)
        revisions = query('SELECT * FROM revisions WHERE image_id=? ORDER BY created', (image_id,))
        for rev in revisions:
            saved = Path('history') / image_id / (rev['id'] + '.png')
            atomic_copy(rev['path'], DELETED / saved)
            rev['path'] = str(saved)
        mask = None
        chosen = row['current'] or row['prediction']
        if chosen:
            rev = next(r for r in revisions if r['id'] == chosen)
            mask = Path('masks') / (image_id + '.png')
            atomic_copy(DELETED / rev['path'], DELETED / mask)
        metadata = {**row, 'image': str(image), 'mask': str(mask) if mask else None,
                    'previous_status': row['status'], 'status': 'deleted',
                    'archived_at': time.time(), 'revisions': revisions}
        metadata_dir = DELETED / 'metadata'
        metadata_dir.mkdir(exist_ok=True)
        temporary = metadata_dir / ('.' + uid())
        temporary.write_text(json.dumps(metadata, indent=2))
        temporary.replace(metadata_dir / (image_id + '.json'))
        update("UPDATE images SET status='deleted',updated=? WHERE id=?", (time.time(), image_id))
        sync_approved()
    return {'ok': True, 'path': str(DELETED), 'image': str(image), 'mask': str(mask) if mask else None}


@app.post('/api/images/{image_id}/apply')
def apply_revision(image_id: str, body: ApplyRequest):
    with LOCK:
        row = item(image_id)
        editable(row)
        rev = query('SELECT * FROM revisions WHERE id=? AND image_id=?', (body.revision, image_id), True)
        if not rev:
            raise HTTPException(404, 'Revision not found')
        adopted = revision(image_id, 'edited', Path(rev['path']), rev['model'])
        update("UPDATE images SET current=?,status='unreviewed',updated=? WHERE id=?", (adopted, time.time(), image_id))
        sync_approved()
    return {'ok': True}


@app.post('/api/models')
def add_model(body: ModelRequest):
    # Local checkpoints only: PyTorch pickle must be from a trusted source.
    from autorace_lane_lite.model import load_checkpoint
    path = Path(body.path).expanduser().resolve()
    try:
        _, checkpoint = load_checkpoint(path)
        if checkpoint['size'] != [640, 480]:
            raise ValueError('640×480 checkpoint required')
    except Exception as e:
        raise HTTPException(400, str(e))
    return register_model(path, body.name or path.stem)


def selected_model(model_id):
    model = query('SELECT * FROM models WHERE id=?', (model_id,), True)
    if not model:
        raise HTTPException(404, 'Model not found')
    return model


def submit(kind, payload):
    job_id = uid()
    update('INSERT INTO jobs VALUES (?,?,?,?,?,?,?)',
           (job_id, kind, 'queued', json.dumps(payload), None, None, time.time()))
    POOL.submit(run_job, job_id, kind, payload)
    return {'job': job_id}


@app.post('/api/infer')
def infer(body: InferRequest):
    selected_model(body.model)
    rows = [item(i) for i in dict.fromkeys(body.images)]
    if any(r['status']=='deleted' for r in rows):
        raise HTTPException(409, 'Deleted images cannot be inferred')
    return submit('infer', {'model': body.model, 'images': [r['id'] for r in rows]})


def atomic_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name('.' + destination.name + '.' + uid())
    try:
        shutil.copy2(source, temp)
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)


def sync_approved():
    """Mirror saved edits/approvals; source images and past snapshots stay intact."""
    with LOCK:
        APPROVED.mkdir(parents=True, exist_ok=True)
        manifest_path = APPROVED / 'manifest.json'
        old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'samples': []}
        previous = {r['id']: r for r in old['samples']}
        rows = eligible_rows()
        registry = dataset_store.read(dataset_root())
        entries = registry['entries'] if registry else {}
        samples = []
        for row in rows:
            incorporated = entries.get(row['origin'])
            if incorporated and incorporated['revision'] == row['current']:
                continue
            image = Path('images') / (row['id'] + Path(row['path']).suffix)
            mask = Path('masks') / (row['id'] + '.png')
            prior = previous.get(row['id'])
            if not (APPROVED / image).is_file():
                atomic_copy(row['path'], APPROVED / image)
            if not prior or prior['revision'] != row['current'] or not (APPROVED / mask).is_file():
                atomic_copy(row['mask_path'], APPROVED / mask)
            samples.append({'id': row['id'], 'image': str(image), 'mask': str(mask),
                            'sha256': row['digest'], 'source': row['name'],
                            'origin': row['origin'], 'capture_session': row['session'],
                            'status': row['status'], 'revision': row['current']})
        metadata = {'classes': ['background', 'orange_lane', 'white_dashed_lane'],
                    'ignore_index': 255, 'samples': samples,
                    'purpose': 'Pending saved/approved revisions; merged into train_dataset on training'}
        temporary = APPROVED / ('.manifest.' + uid())
        temporary.write_text(json.dumps(metadata, indent=2))
        temporary.replace(manifest_path)
        retained = {r['id'] for r in samples}
        for prior in old['samples']:
            if prior['id'] not in retained:
                for key in ['image', 'mask']:
                    path = (APPROVED / prior[key]).resolve()
                    if path.is_relative_to(APPROVED) and path.name.startswith(prior['id']):
                        path.unlink(missing_ok=True)
        publish_layout()
        return metadata


def publish_layout(seed=None):
    with LOCK:
        return _publish_layout(seed)

def _publish_layout(seed=None):
    rows=eligible_rows()
    samples=[{'id':r['id'],'origin':r['origin'],'sha256':r['digest'],'revision':r['current'],'image':r['path'],'label':r['mask_path']} for r in rows]
    unlabeled=[{'id':r['id'],'image':r['path']} for r in query("SELECT * FROM images WHERE status IN ('unreviewed','retry')")]
    return dataset_layout.publish(DATA_ROOT,samples,unlabeled,dataset_root()/'registry.json','masks',seed=seed)


def snapshot(seed=31):
    with LOCK:
        if len(eligible_rows()) < 5:
            raise HTTPException(400, 'At least 5 saved or approved images required')
        registry = merge_pending()
        entries = registry['entries']
        rows = []
        for row in eligible_rows():
            entry = entries.get(row['origin'])
            if not entry or entry['revision'] != row['current']:
                raise ValueError('Incomplete pending merge')
            rows.append(dict(entry))
        evaluation = sorted((r for r in rows if r['group']=='eval'), key=lambda r: r['origin'])
        pool_rows = sorted((r for r in rows if r['group']=='pool'), key=lambda r: r['origin'])
        if len(pool_rows)<2 or not evaluation:
            raise HTTPException(400, 'At least two train/val images and one fixed eval image required')
        random.Random(seed).shuffle(pool_rows)
        train_count = max(1, min(len(pool_rows)-1, round(len(pool_rows)*.75)))
        rows = pool_rows + evaluation
        assignments = ['train']*train_count + ['val']*(len(pool_rows)-train_count) + ['test']*len(evaluation)
        version = uid()
        staging = dataset_root() / 'versions' / ('.' + version)
        staging.mkdir(parents=True)
        samples, counts, digests = [], {}, set()
        try:
            for row, split in zip(rows, assignments):
                source_image = dataset_root() / row['image']
                digest = hashlib.sha256(source_image.read_bytes()).hexdigest()
                if digest != row['sha256'] or digest in digests:
                    raise ValueError('Image changed or duplicate image in approved dataset: ' + row['source'])
                digests.add(digest)
                disk_split = {'train':'train','val':'test','test':'eval'}[split]
                image = Path(disk_split) / 'images' / Path(row['image']).name
                mask = Path(disk_split) / 'masks' / Path(row['mask']).name
                (staging / image).parent.mkdir(parents=True, exist_ok=True)
                (staging / mask).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_image, staging / image)
                shutil.copy2(dataset_root() / row['mask'], staging / mask)
                with Image.open(staging / mask) as im:
                    a = np.array(im)
                with Image.open(staging / image) as im:
                    if a.shape != (im.height, im.width) or not np.isin(a, [0, 1, 2, 255]).all():
                        raise ValueError('Invalid class ID mask: ' + row['source'])
                counts.setdefault(split, [0, 0, 0, 0])
                counts[split] = [x + int((a == c).sum()) for x, c in zip(counts[split], [0, 1, 2, 255])]
                samples.append({**row, 'image': str(image), 'mask': str(mask), 'split': split})
            manifest = {'classes': ['background', 'orange_lane', 'white_dashed_lane'], 'ignore_index': 255, 'samples': samples,
                        'pixel_counts_0_1_2_255': counts,
                        'split_policy': {'policy': dataset_store.POLICY, 'policy_id': registry['id'], 'seed': seed,
                                         'eval_seed': 31, 'unit': 'image', 'source': str(dataset_root()),
                                         'eval_membership': 'permanent; test is eval', 'pool_train_ratio': .75}}
            (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
            destination = staging.with_name(version)
            staging.rename(destination)
        except Exception:
            shutil.rmtree(staging)
            raise
    publish_layout(seed)
    return {'version': version, 'path': str(destination), 'source': str(dataset_root()), 'policy_id': registry['id'],
            'counts': {s: assignments.count(s) for s in ['train', 'val', 'test']}, 'seed': seed}


@app.post('/api/datasets')
def export_dataset():
    return snapshot()


@app.post('/api/train')
def train(body: TrainRequest):
    registry = dataset_store.read(dataset_root())
    if body.model != 'fresh':
        selected_model(body.model)
        record = query('SELECT * FROM model_policies WHERE model_id=?', (body.model,), True)
        if not registry or not record or record['policy_id'] != registry['id']:
            raise HTTPException(409, 'Legacy model: start with ImageNet initial weights to prevent eval leakage')
    data = snapshot(body.seed)
    return submit('train', {**body.model_dump(), 'dataset': data})


def validate_snapshot(data):
    registry = dataset_store.read(dataset_root())
    if not registry or data.get('policy_id') != registry['id']:
        raise ValueError('Dataset policy mismatch')
    manifest = json.loads((Path(data['path'])/'manifest.json').read_text())
    seen = set()
    for row in manifest['samples']:
        entry = registry['entries'].get(row['origin'])
        if not entry or (entry['group']=='eval') != (row['split']=='test'):
            raise ValueError('Eval image crossed the training boundary')
        digest = hashlib.sha256((Path(data['path'])/row['image']).read_bytes()).hexdigest()
        if digest in seen or digest != row['sha256']:
            raise ValueError('Duplicate or changed image in dataset version')
        seen.add(digest)


def run_job(job_id, kind, payload):
    update("UPDATE jobs SET state='running' WHERE id=?", (job_id,))
    try:
        model = selected_model(payload['model']) if payload['model'] != 'fresh' else None
        if kind == 'infer':
            result = inference(job_id, model, payload['images'])
        else:
            output = WORK / 'runs' / job_id
            command = [sys.executable, str(TRAIN / 'train.py'), '--data', payload['dataset']['path'],
                       '--output', str(output), '--size', '640x480',
                       '--epochs', str(payload['epochs']), '--batch-size', str(payload['batch_size']),
                       '--lr', str(payload['lr']), '--seed', str(payload['seed']), '--workers', '2', '--device', os.environ.get('LANE_REVIEW_TRAIN_DEVICE', 'cuda')]
            command += ['--pretrained'] if model is None else ['--weights', model['path']]
            validate_snapshot(payload['dataset'])
            log = WORK / 'logs' / f'{job_id}.log'
            with log.open('w') as stream:
                process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                         env={**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                                              'TORCH_HOME': str(TRAIN.parent / 'models/torch_cache')})
            if process.returncode:
                raise RuntimeError(log.read_text()[-4000:])
            metrics = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
            best = max(metrics, key=lambda r: r['val']['lane_miou'])
            registered = register_model(output / 'best.pt', f"추가 학습 {job_id[:6]} · IoU {best['val']['lane_miou']:.1%}")
            evaluation = [sys.executable, str(TRAIN / 'train.py'), '--data', payload['dataset']['path'],
                          '--size', '640x480', '--weights', str(output / 'best.pt'), '--evaluate', 'test',
                          '--batch-size', str(payload['batch_size']), '--workers', '2', '--device', os.environ.get('LANE_REVIEW_TRAIN_DEVICE', 'cuda')]
            evaluated = subprocess.run(evaluation, capture_output=True, text=True,
                                       env={**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                                              'TORCH_HOME': str(TRAIN.parent / 'models/torch_cache')})
            if evaluated.returncode:
                raise RuntimeError('Test evaluation failed: ' + evaluated.stderr[-4000:])
            test_metrics = json.loads(evaluated.stdout)
            (output / 'test_metrics.json').write_text(json.dumps(test_metrics, indent=2))
            update('INSERT OR REPLACE INTO model_policies VALUES (?,?)', (registered['id'], payload['dataset']['policy_id']))
            result = {'model': registered, 'best': best, 'test': test_metrics,
                      'dataset': payload['dataset'], 'output': str(output)}
        update("UPDATE jobs SET state='complete',result=? WHERE id=?", (json.dumps(result), job_id))
    except Exception as e:
        update("UPDATE jobs SET state='failed',error=? WHERE id=?", (str(e), job_id))


def inference(job_id, model, image_ids):
    import torch
    from autorace_lane_lite.model import load_checkpoint
    from autorace_lane_lite.runtime import preprocess
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network, _ = load_checkpoint(model['path'], device)
    network.eval()
    errors, completed = [], 0
    try:
        for image_id in image_ids:
            try:
                row = item(image_id)
                if row['status'] == 'deleted':
                    continue
                rgb = np.asarray(Image.open(row['path']).convert('RGB'))
                data = preprocess(rgb[..., ::-1].copy(), 640, 480)
                with torch.inference_mode():
                    mask = network(torch.from_numpy(data).to(device)).argmax(1)[0].cpu().numpy().astype(np.uint8)
                mask = Image.fromarray(mask).resize((row['width'], row['height']), Image.Resampling.NEAREST)
                path = WORK / 'masks' / f'{uid()}.png'
                mask.save(path)
                with LOCK:
                    rev = revision(image_id, 'prediction', path, model['id'])
                    # Existing edits/approved labels always remain the current mask.
                    update('UPDATE images SET prediction=?,current=COALESCE(current,?),status=CASE WHEN status=\'retry\' THEN \'unreviewed\' ELSE status END WHERE id=?', (rev, rev, image_id))
                completed += 1
            except Exception as e:
                errors.append({'image': image_id, 'error': str(e)})
                if item(image_id)['split'] == 'train':
                    update("UPDATE images SET status='retry' WHERE id=?", (image_id,))
            update('UPDATE jobs SET result=? WHERE id=?', (json.dumps({'completed': completed, 'total': len(image_ids), 'errors': errors}), job_id))
    finally:
        del network
        if device == 'cuda':
            torch.cuda.empty_cache()
    return {'completed': completed, 'total': len(image_ids), 'errors': errors}


app.mount('/', StaticFiles(directory=HERE / 'static', html=True), name='ui')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=int(os.environ.get('LANE_REVIEW_PORT', '8765')))
