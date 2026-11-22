"""Storage and model integration checks using isolated temporary data."""
import json,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
from pathlib import Path
from fastapi import HTTPException
from PIL import Image
from bbox import app as b
class ReviewTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.old=(b.WORK,b.ROOT);b.WORK=Path(self.tmp.name)/'work';b.ROOT=Path(self.tmp.name)/'data';b.initialize()
  self.src=Path(self.tmp.name)/'input';self.src.mkdir()
  for i in range(8):Image.new('RGB',(96,64),(i*25,60,90)).save(self.src/f'session__frame_{i}.png')
  self.assertEqual(b.import_images(b.Import(path=str(self.src)))['added'],8)
  self.assertEqual(b.import_images(b.Import(path=str(self.src)))['added'],0)
  self.rows=b.state()['images']
  imports=list((b.ROOT/'_store/imports').glob('*.json'));self.assertEqual(len(imports),1)
  record=json.loads(imports[0].read_text());self.assertEqual(record['added'],8)
  self.assertTrue(all(Path(r['unlabeled_image']).is_file() for r in record['images']))
  self.assertTrue(all(r['status']=='unreviewed' and r['current'] is None for r in self.rows))
 def tearDown(self):b.WORK,b.ROOT=self.old;self.tmp.cleanup()
 def save(self,r,boxes=None,expected=None):return b.save(r['id'],b.Save(boxes=boxes or [],expected_revision=expected))
 def test_edit_history_export_and_delete(self):
  box={'id':'box','class_id':0,'x1':4,'y1':5,'x2':40,'y2':30};r=self.rows[0]
  rev=self.save(r,[box])['revision']
  with self.assertRaises(HTTPException) as cm:self.save(r,[box])
  self.assertEqual(cm.exception.status_code,409)
  with self.assertRaises(HTTPException):self.save(r,[{**box,'x2':200}],rev)
  self.assertEqual(b.boxes(rev),[box])
  newer=self.save(r,[],rev)['revision'];self.assertNotEqual(rev,newer);self.assertEqual(b.boxes(rev),[box])
  b.apply(r['id'],b.Apply(revision=rev))
  for r in self.rows[1:6]:self.save(r)
  d=b.snapshot();self.assertEqual(d['counts'],{'train':4,'val':1,'test':1})
  manifest=json.loads((Path(d['path'])/'manifest.json').read_text());groups={r['origin']:r['group'] for r in manifest['samples']}
  for r in self.rows[6:]:self.save(r)
  d2=b.snapshot(19);m2=json.loads((Path(d2['path'])/'manifest.json').read_text())
  self.assertTrue(all(groups.get(r['origin'],r['group'])==r['group'] for r in m2['samples']))
  for r in m2['samples']:
   label=Path(d2['path'])/{'train':'train','val':'test','test':'eval'}[r['split']]/'labels'/f"{r['id']}.txt"
   if r['id']==self.rows[0]['id']:
    values=label.read_text().split();self.assertEqual(int(values[0]),0);self.assertAlmostEqual(float(values[3]),36/96)
  view=json.loads((b.ROOT/'manifest.json').read_text())
  expected={r['id']:{'train':'train','val':'test','test':'eval'}[r['split']] for r in m2['samples']}
  self.assertTrue(all(expected[r['id']]==r['split'] for r in view['samples']))
  for split in ('train','test','eval'):
   self.assertEqual({p.stem for p in (b.ROOT/split/'images').iterdir()},{p.stem for p in (b.ROOT/split/'labels').iterdir()})
  r=self.rows[0];b.delete(r['id'])
  with self.assertRaises(HTTPException):self.save(r)
  self.assertTrue((b.ROOT/'deleted/metadata'/f"{r['id']}.json").exists());self.assertTrue(Path(d['path']).exists())
 def test_empty_and_managed_input(self):
  empty=Path(self.tmp.name)/'empty';empty.mkdir()
  self.assertEqual(b.import_images(b.Import(path=str(empty)))['scanned'],0)
  src=b.ROOT/'unlabeled/new.png';Image.new('RGB',(32,32),(250,250,250)).save(src)
  d=b.import_images(b.Import(path=str(src.parent)))
  self.assertEqual(d['added'],1);self.assertTrue(d['ids'])
  self.assertFalse(src.exists());self.assertTrue((b.ROOT/'_store/source_inputs/new.png').exists())
 def test_inference_retains_edit(self):
  r=self.rows[0];rev=self.save(r)['revision']
  b.run('missing','infer',{'model':'best','images':[r['id']]})
  row=b.item(r['id']);self.assertEqual(row['current'],rev);self.assertIsNotNone(row['prediction']);self.assertEqual(b.boxes(rev),[])
 def test_confidence_validation(self):
  from pydantic import ValidationError
  self.assertEqual(b.Infer(model='best',images=['image']).confidence,.25)
  for value in (.01,.6,1):self.assertEqual(b.Infer(model='best',images=['image'],confidence=value).confidence,value)
  for value in (0,-.1,1.01,float('nan'),float('inf')):
   with self.assertRaises(ValidationError):b.Infer(model='best',images=['image'],confidence=value)
 def test_state_distinguishes_uninferred_empty_and_detected_images(self):
  empty,detected=self.rows[:2]
  box={'id':'box','class_id':0,'x1':4,'y1':5,'x2':40,'y2':30}
  for row,values in ((empty,[]),(detected,[box])):
   rev=b.revision(row['id'],values,'prediction','best')
   b.write('UPDATE images SET prediction=?,current=? WHERE id=?',(rev,rev,row['id']))
  counts={r['id']:r['prediction_box_count'] for r in b.state()['images']}
  self.assertEqual(counts[empty['id']],0)
  self.assertEqual(counts[detected['id']],1)
  self.assertIsNone(counts[self.rows[2]['id']])
  rev=b.revision(detected['id'],[],'prediction','best')
  b.write('UPDATE images SET prediction=? WHERE id=?',(rev,detected['id']))
  counts={r['id']:r['prediction_box_count'] for r in b.state()['images']}
  self.assertEqual(counts[detected['id']],0)
 def test_saved_label_counts_use_current_revision_and_update_after_edit(self):
  row=self.rows[0];box={'id':'first','class_id':0,'x1':4,'y1':5,'x2':40,'y2':30}
  pred=b.revision(row['id'],[box],'prediction','best')
  b.write('UPDATE images SET prediction=? WHERE id=?',(pred,row['id']))
  edited=[{**box,'class_id':2},{**box,'id':'second','class_id':2},{**box,'id':'third','class_id':5}]
  rev=self.save(row,edited)['revision']
  b.status(row['id'],b.Status(status='approved'))
  result=next(r for r in b.state()['images'] if r['id']==row['id'])
  self.assertEqual(result['prediction_box_count'],1)
  self.assertEqual(result['label_box_count'],3)
  self.assertEqual(result['label_class_counts'],[0,0,2,0,0,1,0,0])
  self.save(row,[],rev)
  result=next(r for r in b.state()['images'] if r['id']==row['id'])
  self.assertEqual(result['label_box_count'],0)
  self.assertEqual(result['label_class_counts'],[0]*len(b.NAMES))
 def test_inference_uses_selected_confidence_and_legacy_default(self):
  empty=Mock();empty.xyxy.tolist.return_value=[];empty.cls.tolist.return_value=[];empty.conf.tolist.return_value=[]
  detector=Mock();detector.predict.return_value=[SimpleNamespace(boxes=empty)]
  r=self.rows[0]
  for setting,expected in (({'confidence':.6},.6),({},.25)):
   with patch.object(b,'load',return_value=detector):
    b.run('missing','infer',{'model':'best','images':[r['id']],**setting})
   detector.predict.assert_called_with(r['path'],imgsz=640,conf=expected,verbose=False)
   self.assertIsNotNone(b.item(r['id'])['prediction'])
class ModelCompatibilityTest(unittest.TestCase):
 def test_head_expansion_preserves_named_classes(self):
  import torch
  from bbox.engine import load,reorder,NAMES
  source=Path(__file__).parent/'models/best.pt'
  original=load(source)
  with tempfile.TemporaryDirectory() as temp:
   converted=load(reorder(source,Path(temp)/'expanded.pt'))
   self.assertEqual(list(converted.names.values()),NAMES)
   self.assertEqual(converted.model.yaml['nc'],len(NAMES))
   head=converted.model.model[-1]
   self.assertEqual(head.nc,len(NAMES))
   for attr in ('cv3','one2one_cv3'):
    for old,new in zip(getattr(original.model.model[-1],attr),getattr(head,attr)):
     self.assertEqual(new[-1].out_channels,len(NAMES))
     for old_id,name in original.names.items():
      new_id=NAMES.index(name)
      self.assertTrue(torch.equal(old[-1].weight[old_id],new[-1].weight[new_id]))
      self.assertTrue(torch.equal(old[-1].bias[old_id],new[-1].bias[new_id]))
   import numpy as np
   result=converted.predict(np.zeros((480,640,3),dtype=np.uint8),device='cpu',verbose=False)[0]
   self.assertEqual(result.names,dict(enumerate(NAMES)))

class RoutingTest(unittest.IsolatedAsyncioTestCase):
 async def test_mounted_pages_api_and_origin(self):
  import asyncio,httpx,server
  async def heartbeat():
   while True:await asyncio.sleep(.01)
  # Keep event-loop wakeups working in restricted socket environments too.
  tick=asyncio.create_task(heartbeat())
  try:
   async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),base_url='http://testserver',follow_redirects=True) as c:
    for path in ('/','/segmentation/','/bbox/','/bbox/app.js','/segmentation/app.js'):
     r=await asyncio.wait_for(c.get(path),5);self.assertEqual(r.status_code,200)
    r=await c.post('/bbox/api/import',json={'path':'/missing'},headers={'origin':'http://foreign'})
    self.assertEqual(r.status_code,403)
  finally:tick.cancel()
if __name__=='__main__':unittest.main()
