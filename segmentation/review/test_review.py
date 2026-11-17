"""Actual CUDA inference, clean reset, incremental training, and permanent eval checks."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
import app as review


class ReviewFlow(unittest.TestCase):
    def test_fixed_eval_lifecycle(self):
        with tempfile.TemporaryDirectory() as temp:
            original = review.WORK, review.BASE, review.APPROVED, review.DELETED
            old_work, old_base, _, _ = original
            review.WORK = Path(temp)/'work'
            review.BASE = Path(temp)/'base'; review.BASE.mkdir()
            review.APPROVED = Path(temp)/'approve_dataset'
            review.DELETED = Path(temp)/'delete_dataset'
            source = json.loads((old_base/'manifest.json').read_text())['samples']
            initial_rows = [dict(r, image=str(old_base/r['image']), mask=str(old_base/r['mask'])) for r in source[:10]]
            (review.BASE/'manifest.json').write_text(json.dumps({'samples': initial_rows}))
            original_model_hash = hashlib.sha256(review.INITIAL.read_bytes()).hexdigest()
            try:
                with TestClient(review.app) as client:
                    def get(): return client.get('/api/state').json()
                    def post(path, data):
                        r=client.post('/api'+path,json=data)
                        self.assertEqual(r.status_code,200,r.text)
                        return r.json()
                    def wait(job):
                        deadline=time.time()+120
                        while time.time()<deadline:
                            row=next(j for j in get()['jobs'] if j['id']==job)
                            if row['state'] in ['complete','failed']:
                                self.assertEqual(row['state'],'complete',row);return row
                            time.sleep(.2)
                        self.fail('Job timeout')
                    def manifest(version):return json.loads(Path(version['path'],'manifest.json').read_text())
                    def eval_keys(version):return {r['origin'] for r in manifest(version)['samples'] if r['split']=='test'}
                    state=get();legacy=state['models'][0]['id']
                    self.assertEqual(state['dataset']['pending'],0)
                    self.assertFalse(state['models'][0]['train_eligible'])
                    self.assertTrue((review.dataset_root()/'migration_backup/review.sqlite').is_file())
                    first=post('/datasets',{})
                    self.assertEqual(first['counts'],{'train':6,'val':2,'test':2})
                    stable=eval_keys(first)
                    changed=review.snapshot(97)
                    self.assertEqual(stable,eval_keys(changed))
                    self.assertEqual(client.post('/api/train',json={'model':legacy,'epochs':1}).status_code,409)
                    pool_origin=next(r['origin'] for r in manifest(first)['samples'] if r['split']=='train')
                    row=next(i for i in get()['images'] if i['origin']==pool_origin)
                    a=np.zeros((row['height'],row['width']),np.uint8);a[100:200,100:110]=1
                    buffer=io.BytesIO();Image.fromarray(a).save(buffer,format='PNG')
                    payload={'mask':base64.b64encode(buffer.getvalue()).decode(),'expected_revision':row['current']}
                    rev=post('/images/'+row['id']+'/mask',payload)['revision']
                    self.assertEqual(get()['dataset']['pending'],1)
                    self.assertEqual(client.post('/api/images/'+row['id']+'/mask',json=payload).status_code,409)
                    np.testing.assert_array_equal(np.array(Image.open(review.APPROVED/'masks'/f"{row['id']}.png")),a)
                    fresh=wait(post('/train',{'model':'fresh','epochs':1,'batch_size':2,'lr':.001})['job'])
                    self.assertEqual(fresh['result']['dataset']['counts'],{'train':6,'val':2,'test':2})
                    self.assertIn('lane_miou',fresh['result']['test'])
                    clean=fresh['result']['model']['id']
                    self.assertTrue(next(m for m in get()['models'] if m['id']==clean)['train_eligible'])
                    self.assertEqual(get()['dataset']['pending'],0)
                    # Add five new labels; old eval never changes even when the seed changes.
                    with review.LOCK:
                        for r in source[10:15]:
                            review.add_image(old_base/r['image'],Path(r['source']).name,r.get('capture_session','new'),old_base/r['mask'])
                        review.sync_approved()
                    self.assertEqual(get()['dataset']['pending'],5)
                    expanded=review.snapshot(7)
                    self.assertEqual(expanded['counts'],{'train':9,'val':3,'test':3})
                    self.assertTrue(stable<=eval_keys(expanded))
                    earlier_train={r['origin'] for r in manifest(fresh['result']['dataset'])['samples'] if r['split']=='train'}
                    self.assertFalse(earlier_train & eval_keys(expanded))
                    incremental=wait(post('/train',{'model':clean,'epochs':1,'batch_size':2,'seed':97})['job'])
                    self.assertEqual(eval_keys(expanded),eval_keys(incremental['result']['dataset']))
                    # Actual inference does not replace human edits.
                    before=next(i for i in get()['images'] if i['id']==row['id'])
                    inferred=wait(post('/infer',{'model':clean,'images':[row['id']]})['job'])
                    self.assertFalse(inferred['result']['errors'])
                    after=next(i for i in get()['images'] if i['id']==row['id'])
                    self.assertEqual(before['current'],after['current'])
                    # Eval label revisions retain membership; deleting it retains the tombstone.
                    eval_row=next(i for i in get()['images'] if i['origin'] in stable)
                    payload['expected_revision']=eval_row['current']
                    post('/images/'+eval_row['id']+'/mask',payload)
                    updated=review.snapshot(111)
                    self.assertEqual(eval_keys(expanded),eval_keys(updated))
                    archive=post('/images/'+eval_row['id']+'/delete',{})
                    self.assertTrue((review.DELETED/archive['image']).is_file())
                    self.assertTrue((review.DELETED/archive['mask']).is_file())
                    self.assertEqual(review.dataset_store.read(review.dataset_root())['entries'][eval_row['origin']]['group'],'eval')
                    self.assertEqual(client.post('/api/infer',json={'model':clean,'images':[eval_row['id']]}).status_code,409)
                    review.initialize()
                    restarted=review.snapshot(999)
                    self.assertEqual(eval_keys(updated)-{eval_row['origin']},eval_keys(restarted))
                    self.assertEqual(next(i for i in get()['images'] if i['id']==eval_row['id'])['status'],'deleted')
                    # A malicious or accidental split change is rejected before training.
                    f=Path(restarted['path'])/'manifest.json';d=json.loads(f.read_text())
                    next(r for r in d['samples'] if r['split']=='test')['split']='train'
                    f.write_text(json.dumps(d))
                    with self.assertRaises(ValueError):review.validate_snapshot(restarted)
                    self.assertEqual(original_model_hash,hashlib.sha256(review.INITIAL.read_bytes()).hexdigest())
                    self.assertFalse(list(review.WORK.rglob('*.onnx')))
            finally:
                review.WORK, review.BASE, review.APPROVED, review.DELETED=original

if __name__=='__main__':unittest.main(verbosity=2)
