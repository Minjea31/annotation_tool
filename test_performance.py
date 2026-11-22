"""Regression checks for shared-state caching and incremental dataset updates."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request

from annotation_state import StateCache
from bbox import app as bbox
import dataset_layout


class StateCacheTest(unittest.TestCase):
    def test_idle_polls_share_json_and_invalidate_on_db_registry_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            database = work / 'review.sqlite'
            registry = work / 'registry.json'
            metric = work / 'runs/train/metrics.jsonl'
            metric.parent.mkdir(parents=True)
            metric.write_text('first')
            with closing(sqlite3.connect(database)) as connection, connection:
                connection.execute('CREATE TABLE data(value INTEGER)')
            cache = StateCache()
            calls = []

            def build():
                calls.append(True)
                return {'images': [], 'jobs': [{'id': 'train', 'kind': 'train', 'state': 'complete'}],
                        'metric': metric.read_text(), 'count': len(calls)}

            def get(etag=None, query=b''):
                headers = [] if etag is None else [(b'if-none-match', etag.encode())]
                request = Request({'type': 'http', 'method': 'GET', 'path': '/api/state',
                                   'query_string': query, 'headers': headers})
                return cache.response(request, work, registry, build)

            first = get()
            tag = first.headers['etag']
            with ThreadPoolExecutor(max_workers=8) as pool:
                responses = list(pool.map(lambda _: get(tag), range(16)))
            self.assertTrue(all(r.status_code == 304 and r.body == b'' for r in responses))
            self.assertEqual(len(calls), 1)
            with closing(sqlite3.connect(database)) as connection, connection:
                connection.execute('INSERT INTO data VALUES(1)')
            changed = get(tag)
            self.assertEqual(changed.status_code, 200)
            self.assertEqual(len(calls), 2)
            registry.write_text('{}')
            changed = get(changed.headers['etag'])
            self.assertEqual(changed.status_code, 200)
            metric.write_text('second metric')
            changed = get(changed.headers['etag'])
            self.assertEqual(changed.status_code, 200)
            self.assertEqual(json.loads(changed.body)['metric'], 'second metric')
            for index in range(25):
                get(query=f'image_id={index}'.encode())
            self.assertLessEqual(len(cache.entries), 16)

    def test_running_training_never_reuses_a_metric_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = StateCache()
            request = Request({'type': 'http', 'path': '/', 'headers': [], 'query_string': b''})
            build = lambda: {'jobs': [{'id': 'train', 'kind': 'train', 'state': 'running'}]}
            response = cache.response(request, root, root/'registry.json', build)
            self.assertNotIn('etag', response.headers)
            self.assertFalse(cache.entries)


class IncrementalSaveTest(unittest.TestCase):
    def test_approval_does_not_recopy_other_images_and_stale_write_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            original = bbox.ROOT, bbox.WORK
            bbox.ROOT = Path(directory)/'data'
            bbox.WORK = bbox.ROOT/'_store/review'
            try:
                bbox.initialize()
                from PIL import Image
                source = Path(directory)/'input'
                source.mkdir()
                for index in range(8):
                    Image.new('RGB', (96, 64), (index*30, 50, 70)).save(source/f'{index}.png')
                bbox.import_images(bbox.Import(path=str(source)))
                rows = bbox.state()['images']
                revisions = {}
                for row in rows:
                    revisions[row['id']] = bbox.save(row['id'], bbox.Save(boxes=[]))['revision']
                target = rows[0]['id']
                retained = json.loads((bbox.ROOT/'manifest.json').read_text())
                assignments = {row['id']: row['split'] for row in retained['samples']}
                before = {row['id']: (bbox.ROOT/row['label']).stat().st_ino
                          for row in retained['samples'] if row['id'] != target}
                with patch.object(bbox, 'copy', wraps=bbox.copy) as copy, \
                     patch.object(dataset_layout, 'link', wraps=dataset_layout.link) as link:
                    result = bbox.save(target, bbox.Save(boxes=[], expected_revision=revisions[target], approve=True))
                    self.assertEqual(result['image']['status'], 'approved')
                    self.assertEqual(copy.call_count, 0)
                    self.assertEqual(link.call_count, 2)
                after = json.loads((bbox.ROOT/'manifest.json').read_text())
                self.assertEqual(assignments, {row['id']: row['split'] for row in after['samples']})
                for row in after['samples']:
                    if row['id'] in before:
                        self.assertEqual((bbox.ROOT/row['label']).stat().st_ino, before[row['id']])
                with self.assertRaises(HTTPException) as conflict:
                    bbox.status(target, bbox.Status(status='approved', expected_revision=revisions[target]))
                self.assertEqual(conflict.exception.status_code, 409)
                bbox.status(target, bbox.Status(status='excluded', expected_revision=result['revision']))
                manifest = json.loads((bbox.ROOT/'manifest.json').read_text())
                self.assertNotIn(target, {row['id'] for row in manifest['samples']})
                # A full publish still repairs files removed outside the tool.
                remaining = manifest['samples'][0]
                missing = bbox.ROOT/remaining['label']
                missing.unlink()
                bbox.publish_layout()
                self.assertTrue(missing.is_file())
            finally:
                bbox.ROOT, bbox.WORK = original


if __name__ == '__main__':
    unittest.main()
