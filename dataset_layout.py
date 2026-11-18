"""Publish paired datasets while preserving fixed evaluation membership."""
import hashlib,json,os,shutil,uuid
from pathlib import Path

def atomic(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name('.'+uuid.uuid4().hex);tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
def link(source,destination):
 source=Path(source);destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
 if destination.exists() and os.path.samefile(source,destination):return
 tmp=destination.with_name('.'+uuid.uuid4().hex)
 try:
  try:os.link(source,tmp)
  except OSError:shutil.copy2(source,tmp)
  tmp.replace(destination)
 finally:tmp.unlink(missing_ok=True)
def split_assignments(samples,entries,seed):
 """Keep imported train/validation membership and fixed eval groups."""
 import random
 pool=sorted([r for r in samples if entries[r['origin']]['group']=='pool'],key=lambda r:r['origin'])
 random.Random(seed).shuffle(pool)
 cut=max(1,min(len(pool)-1,round(len(pool)*.75))) if len(pool)>1 else len(pool)
 assignments={r['id']:'eval' for r in samples if entries[r['origin']]['group']=='eval'}
 pinned={r['id']:entries[r['origin']]['view_split'] for r in pool if entries[r['origin']].get('view_split') in ('train','test')}
 assignments.update(pinned)
 remaining_train=max(0,cut-sum(split=='train' for split in pinned.values()))
 for i,r in enumerate(r for r in pool if r['id'] not in pinned):
  assignments[r['id']]='train' if i<remaining_train else 'test'
 return assignments
def publish(root,samples,unlabeled,registry_path,label_dir,seed=None):
 """samples: id/origin/sha256/revision/image/label. Only owned view files are cleaned."""
 root=Path(root);registry_path=Path(registry_path)
 registry=json.loads(registry_path.read_text()) if registry_path.exists() else {'policy':'fixed_eval_v1','id':uuid.uuid4().hex,'eval_seed':31,'entries':{},'hash_groups':{}}
 if seed is None:seed=registry.get('view_seed',31)
 registry['view_seed']=seed
 entries=registry['entries'];groups=registry.setdefault('hash_groups',{e['sha256']:e['group'] for e in entries.values()})
 unseen=sorted([r for r in samples if r['origin'] not in entries],key=lambda r:hashlib.sha256(('31:'+r['origin']).encode()).hexdigest())
 needed=max(0,round((len(entries)+len(unseen))*.2)-sum(e['group']=='eval' for e in entries.values()))
 for r in unseen:
  group=groups.get(r['sha256'])
  if group is None:group='eval' if needed>0 else 'pool';needed-=int(group=='eval')
  entries[r['origin']]={'group':group,'sha256':r['sha256'],'revision':None};groups[r['sha256']]=group
 if samples or registry_path.exists():atomic(registry_path,registry)
 assignments=split_assignments(samples,entries,seed)
 manifest_path=root/'manifest.json';old=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'files':[]}
 owned=[];rows=[]
 for split in ('train','test','eval'):
  (root/split/'images').mkdir(parents=True,exist_ok=True);(root/split/label_dir).mkdir(exist_ok=True)
 (root/'unlabeled').mkdir(parents=True,exist_ok=True)
 for r in samples:
  split=assignments[r['id']];image=Path(split)/'images'/(r['id']+Path(r['image']).suffix);label=Path(split)/label_dir/(r['id']+Path(r['label']).suffix)
  link(r['image'],root/image);link(r['label'],root/label);owned.extend([str(image),str(label)])
  rows.append({**{k:v for k,v in r.items() if k not in ('image','label')},'split':split,'image':str(image),'label':str(label)})
 for r in unlabeled:
  image=Path('unlabeled')/(r['id']+Path(r['image']).suffix);link(r['image'],root/image);owned.append(str(image))
 keep=set(owned)
 for relative in old['files']:
  if relative not in keep:
   p=(root/relative).resolve()
   if p.is_relative_to(root.resolve()):p.unlink(missing_ok=True)
 atomic(manifest_path,{'splits':{'train':'training','test':'validation','eval':'fixed final evaluation'},'files':owned,'samples':rows})
 return rows

def archive_imports(source,work,root,rows):
 """An input placed in managed unlabeled has a durable archive before moving out."""
 source=Path(source).resolve();root=Path(root).resolve()
 if source!=root/'unlabeled':return
 for r in rows:
  original=source/r['name']
  if original.is_file() and original.resolve()!=Path(r['path']).resolve():
   if hashlib.sha256(original.read_bytes()).hexdigest()==r['digest']:
    dest=root/'_store/source_inputs'/original.name
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()!=r['digest']:dest=dest.with_name(r['id']+dest.suffix)
    link(original,dest);original.unlink()

def record_import(root,source,rows,duplicates=0,errors=None):
 """Record a dataset addition; owned images are already copied into review storage."""
 import time
 root=Path(root);source=Path(source)
 if not rows:return None
 batch=uuid.uuid4().hex
 data={'id':batch,'name':source.name,'source':str(source),'created':time.time(),
       'destination':str(root/'unlabeled'),'added':len(rows),'duplicates':duplicates,'errors':errors or [],
       'images':[{'id':r['id'],'name':r['name'],'sha256':r['digest'],
                  'managed_image':r['path'],'unlabeled_image':str(root/'unlabeled'/(r['id']+Path(r['path']).suffix))} for r in rows]}
 atomic(root/'_store/imports'/f'{batch}.json',data)
 return {k:v for k,v in data.items() if k!='images'}
