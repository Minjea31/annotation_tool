"""Two browsers share real storage without running inference or training."""
import json
import gzip
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]


class CollaborationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='annotation-collaboration-')
        cls.directory = Path(cls.temp.name)
        with socket.socket() as connection:
            connection.bind(('127.0.0.1', 0))
            port = connection.getsockname()[1]
        cls.url = f'http://127.0.0.1:{port}'
        env = dict(os.environ, ANNOTATION_HOST='127.0.0.1', ANNOTATION_PORT=str(port),
                   ANNOTATION_SEG_DATA=str(cls.directory/'segmentation'),
                   BBOX_REVIEW_DATA=str(cls.directory/'bbox'),
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        for key in ('LANE_REVIEW_WORK', 'LANE_REVIEW_APPROVE', 'LANE_REVIEW_DELETE',
                    'LANE_REVIEW_DATASET', 'BBOX_REVIEW_WORK'):
            env.pop(key, None)
        cls.log = (cls.directory/'server.log').open('w+')
        cls.process = subprocess.Popen([sys.executable, str(ROOT/'server.py')],
                                       cwd=ROOT, env=env, stdout=cls.log, stderr=cls.log)
        cls.addClassCleanup(cls.cleanup_server)
        for _ in range(100):
            try:
                cls.request('/state')
                break
            except OSError:
                if cls.process.poll() is not None:
                    cls.log.seek(0)
                    raise RuntimeError(cls.log.read())
                time.sleep(.1)
        else:
            raise RuntimeError('Server startup timed out')
        source = cls.directory/'input'
        source.mkdir()
        for index in range(6):
            Image.new('RGB', (96, 64), (index*40, 70, 90)).save(source/f'{index}.png')
        cls.request('/import', {'path': str(source)})
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(
            executable_path='/usr/bin/google-chrome', headless=True, args=['--no-sandbox'])
        cls.addClassCleanup(cls.playwright.stop)
        cls.addClassCleanup(cls.browser.close)

    @classmethod
    def cleanup_server(cls):
        cls.process.terminate()
        try:
            cls.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cls.process.kill()
            cls.process.wait()
        cls.log.close()
        cls.temp.cleanup()

    @classmethod
    def request(cls, path, payload=None):
        req = urllib.request.Request(cls.url+'/segmentation/api'+path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json', 'Origin': cls.url})
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.load(response)

    def setUp(self):
        self.contexts = [self.browser.new_context(viewport={'width':1560, 'height':1050}) for _ in range(2)]
        self.pages = [context.new_page() for context in self.contexts]
        for page in self.pages:
            page.goto(self.url+'/segmentation/')
            page.wait_for_function('image!==null')
            page.select_option('#filter', 'all')
            page.locator('#canvas').focus()

    def tearDown(self):
        for context in self.contexts:
            context.close()

    def paint(self, page, value):
        page.evaluate('(value)=>{mask[0]=value;changed();}', value)

    def test_conditional_state_revalidates_after_each_tools_save(self):
        source = self.directory/'input'
        for prefix in ('/segmentation', '/bbox'):
            def api(path, body=None, etag=None):
                headers = {'Content-Type':'application/json'}
                if etag:
                    headers['If-None-Match'] = etag
                request = urllib.request.Request(self.url+prefix+'/api'+path,
                    data=None if body is None else json.dumps(body).encode(), headers=headers)
                return urllib.request.urlopen(request, timeout=10)
            if prefix == '/bbox':
                api('/import', {'path':str(source)}).close()
            with api('/state') as response:
                state = json.load(response)
                tag = response.headers['ETag']
            with self.assertRaises(urllib.error.HTTPError) as unchanged:
                api('/state', etag=tag)
            self.assertEqual(unchanged.exception.code, 304)
            row = state['images'][0]
            if prefix == '/bbox':
                api('/images/'+row['id']+'/boxes',
                    {'boxes':[], 'expected_revision':row['current'], 'approve':True}).close()
            else:
                api('/images/'+row['id']+'/status',
                    {'status':'retry', 'expected_revision':row['current']}).close()
            with api('/state', etag=tag) as response:
                self.assertEqual(response.status, 200)
                self.assertNotEqual(response.headers['ETag'], tag)

    def test_bbox_approval_uses_one_write_without_waiting_for_state(self):
        request = urllib.request.Request(self.url+'/bbox/api/import',
            data=json.dumps({'path':str(self.directory/'input')}).encode(),
            headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request, timeout=10):
            pass
        first, second = self.pages
        for page in self.pages:
            page.goto(self.url+'/bbox/')
            page.wait_for_function('picture!==null')
            page.select_option('#filter', 'all')
        second.keyboard.press('KeyE')
        second.wait_for_function('(id)=>current.id!==id', arg=first.evaluate('current.id'))
        ids = [page.evaluate('current.id') for page in self.pages]
        writes = [[], []]
        for index, page in enumerate(self.pages):
            page.route('**/api/state*', lambda route: route.abort())
            page.on('request', lambda request, index=index:
                    writes[index].append(request.url) if request.method=='POST' else None)
            page.keyboard.press('KeyW')
        for index, (page, image_id) in enumerate(zip(self.pages, ids)):
            page.wait_for_function('(id)=>current.id!==id&&!busy', arg=image_id)
            self.assertEqual(writes[index], [self.url+'/bbox/api/images/'+image_id+'/boxes'])
        with urllib.request.urlopen(self.url+'/bbox/api/state', timeout=10) as response:
            rows = {row['id']:row for row in json.load(response)['images']}
        self.assertTrue(all(rows[image_id]['status']=='approved' for image_id in ids))

    def test_remote_saved_mask_refreshes_without_page_reload(self):
        first, second = self.pages
        image_id = first.evaluate('current.id')
        self.paint(first, 1)
        first.keyboard.press('KeyW')
        first.wait_for_function('(id)=>current.id!==id', arg=image_id)
        second.evaluate('refresh()')
        self.assertEqual(second.evaluate('mask[0]'), 1)
        self.assertEqual(second.evaluate('current.status'), 'approved')
        self.assertEqual(second.evaluate('activeRevision'), second.evaluate('current.current'))

    def test_different_images_can_be_approved_at_the_same_time(self):
        first, second = self.pages
        second.keyboard.press('KeyE')
        second.wait_for_function('(id)=>current.id!==id', arg=first.evaluate('current.id'))
        ids = [page.evaluate('current.id') for page in self.pages]
        for page in self.pages:
            self.paint(page, 2)
            page.keyboard.press('KeyW')
        for page, image_id in zip(self.pages, ids):
            page.wait_for_function('(id)=>current.id!==id&&!busy', arg=image_id)
        rows = {row['id']:row for row in self.request('/state')['images']}
        self.assertTrue(all(rows[image_id]['status']=='approved' for image_id in ids))

    def test_stale_approval_requires_reviewing_the_changed_label(self):
        first, second = self.pages
        image_id = first.evaluate('current.id')
        self.paint(first, 1)
        first.keyboard.press('KeyV')
        first.wait_for_function('!dirty')
        second.evaluate('refresh()')
        old_revision = second.evaluate('activeRevision')
        frozen = self.request('/state')
        second.route('**/api/state*', lambda route: route.fulfill(json=frozen))
        self.paint(first, 2)
        first.keyboard.press('KeyV')
        first.wait_for_function('(revision)=>!dirty&&activeRevision!==revision', arg=old_revision)
        second.keyboard.press('KeyW')
        second.wait_for_function("document.querySelector('#toast').textContent.includes('다른 사람')")
        row = next(row for row in self.request('/state')['images'] if row['id']==image_id)
        self.assertEqual(row['status'], 'editing')
        self.assertNotEqual(row['current'], old_revision)
        self.assertEqual(second.evaluate('current.id'), image_id)

    def test_approval_uses_one_write_and_skips_the_global_download(self):
        page = self.pages[0]
        image_id = page.evaluate('current.id')
        # A slow or unavailable library refresh must not delay a saved approval.
        page.route('**/api/state*', lambda route: route.abort())
        requests = []
        page.on('request', lambda request: requests.append((request.method, request.url)))
        self.paint(page, 1)
        page.keyboard.press('KeyW')
        page.wait_for_function('(id)=>current.id!==id&&!busy', arg=image_id, timeout=5000)
        writes = [url for method, url in requests if method=='POST']
        self.assertEqual(writes, [self.url+'/segmentation/api/images/'+image_id+'/mask'])
        row = next(row for row in self.request('/state')['images'] if row['id']==image_id)
        self.assertEqual(row['status'], 'approved')

    def test_compact_poll_preserves_progress_and_supports_compression(self):
        image_id = self.pages[0].evaluate('current.id')
        database = self.directory/'segmentation/_store/review/review.sqlite'
        with sqlite3.connect(database) as connection:
            connection.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?)',
                ('compact-test', 'infer', 'complete', json.dumps({'images':[image_id]*3000}),
                 json.dumps({'completed':3000}), None, time.time()))
        try:
            compact = self.request('/state?compact=true&image_id='+image_id)
            full = self.request('/state')
            job = next(job for job in compact['jobs'] if job['id']=='compact-test')
            self.assertEqual(job['payload']['image_count'], 3000)
            self.assertEqual(job['payload']['current_image'], image_id)
            self.assertNotIn('images', job['payload'])
            self.assertNotIn('path', compact['images'][0])
            self.assertLess(len(json.dumps(compact)), len(json.dumps(full))/4)
            req = urllib.request.Request(self.url+'/segmentation/api/state?compact=true',
                                         headers={'Accept-Encoding':'gzip'})
            with urllib.request.urlopen(req) as response:
                self.assertEqual(response.headers.get('Content-Encoding'), 'gzip')
                self.assertEqual(len(json.loads(gzip.decompress(response.read()))['images']), 6)
        finally:
            with sqlite3.connect(database) as connection:
                connection.execute('DELETE FROM jobs WHERE id=?', ('compact-test',))

    def test_dirty_conflict_preserves_edits_and_has_recovery(self):
        first, second = self.pages
        image_id = first.evaluate('current.id')
        self.paint(first, 1)
        self.paint(second, 2)
        first.keyboard.press('KeyW')
        first.wait_for_function('(id)=>current.id!==id', arg=image_id)
        second.evaluate('refresh()')
        self.assertEqual(second.evaluate('mask[0]'), 2)
        self.assertTrue(second.evaluate('dirty'))
        second.keyboard.press('KeyW')
        second.wait_for_function("document.querySelector('#toast').textContent.includes('다른 사람')")
        self.assertEqual(second.evaluate('current.id'), image_id)
        self.assertEqual(second.evaluate('mask[0]'), 2)
        self.assertTrue(second.locator('#syncNotice').is_visible())
        # Cancel keeps the user's unsaved work; accepting loads the shared label.
        second.once('dialog', lambda dialog: dialog.dismiss())
        second.locator('#reloadLabelBtn').click()
        self.assertEqual(second.evaluate('mask[0]'), 2)
        second.once('dialog', lambda dialog: dialog.accept())
        second.locator('#reloadLabelBtn').click()
        second.wait_for_function('!busy&&!dirty')
        self.assertEqual(second.evaluate('mask[0]'), 1)
        self.paint(second, 2)
        second.locator('#approveBtn').click()
        second.wait_for_function('(id)=>current.id!==id', arg=image_id)
        row = next(row for row in self.request('/state')['images'] if row['id']==image_id)
        self.assertEqual(row['status'], 'approved')
        revisions = self.request('/images/'+image_id+'/revisions')
        stored = Path(next(revision['path'] for revision in revisions if revision['id']==row['current']))
        self.assertTrue(stored.is_relative_to(self.directory))
        with Image.open(stored) as mask:
            self.assertEqual(mask.getpixel((0, 0)), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
