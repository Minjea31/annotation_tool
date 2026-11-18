"""Independent bbox review storage; original segmentation data is never imported here."""
import hashlib, json, math, os, random, re, shutil, sqlite3, subprocess, sys, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
from .engine import NAMES, load
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import dataset_layout
HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('BBOX_REVIEW_DATA', HERE/'dataset')).resolve()
WORK=Path(os.environ.get('BBOX_REVIEW_WORK', ROOT/'_store/review')).resolve()
LOCK=threading.RLock(); POOL=ThreadPoolExecutor(max_workers=1)
app=FastAPI()
def uid():return uuid.uuid4().hex
def query(sql,args=(),one=False):
    with sqlite3.connect(WORK/'review.sqlite',timeout=30) as c:
        c.row_factory=sqlite3.Row
        rows=[dict(r) for r in c.execute(sql,args).fetchall()]
    return (rows[0] if rows else None) if one else rows

def write(sql,args=()):
    with sqlite3.connect(WORK/'review.sqlite',timeout=30) as c:c.execute(sql,args)
def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_name('.'+uid()); temp.write_text(json.dumps(data,indent=2));temp.replace(path)
def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True);temp=dst.with_name('.'+uid());shutil.copy2(src,temp);temp.replace(dst)
def initialize():
    WORK.mkdir(parents=True,exist_ok=True)
    for d in ('images','labels','runs','logs'):(WORK/d).mkdir(exist_ok=True)
    with sqlite3.connect(WORK/'review.sqlite') as c:
        c.executescript('''CREATE TABLE IF NOT EXISTS images(id TEXT PRIMARY KEY,name TEXT,path TEXT,digest TEXT UNIQUE,origin TEXT UNIQUE,session TEXT,width INT,height INT,status TEXT,current TEXT,prediction TEXT);
        CREATE TABLE IF NOT EXISTS revisions(id TEXT PRIMARY KEY,image_id TEXT,kind TEXT,path TEXT,model TEXT,created REAL);
        CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY,name TEXT,path TEXT UNIQUE);
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,kind TEXT,state TEXT,payload TEXT,result TEXT,error TEXT,created REAL);''')
        c.execute("UPDATE jobs SET state='interrupted' WHERE state IN ('running','queued')")
    for d in ('images','labels','history','metadata'):(ROOT/'deleted'/d).mkdir(parents=True,exist_ok=True)
    for model_id,name,filename in (
        ('best','best.pt · YOLO26 Nano · 전체 클래스','best.pt'),
        ('red_green_200ep','YOLO26 Nano · red / green · 200 epochs','yolo26n_red_green_200ep.pt'),
    ):
        path=HERE/'models'/filename
        if path.is_file() and not query('SELECT id FROM models WHERE path=?',(str(path),),True):
            write('INSERT INTO models VALUES(?,?,?)',(model_id,name,str(path)))
def item(i):
    r=query('SELECT * FROM images WHERE id=?',(i,),True)
    if not r:raise HTTPException(404,'Image not found')
    return r
def editable(r):
    if r['status']=='deleted':raise HTTPException(409,'삭제 보관된 이미지입니다')
def boxes(rev):
    r=query('SELECT * FROM revisions WHERE id=?',(rev,),True)
    if rev and not r:raise HTTPException(404,'Revision not found')
    return json.loads(Path(r['path']).read_text()) if r else []
@lru_cache(maxsize=8192)
def revision_class_counts(rev):
    counts=[0]*len(NAMES)
    for box in boxes(rev):counts[box['class_id']]+=1
    return tuple(counts)
def prediction_box_count(rev):
    return sum(revision_class_counts(rev))
def revision(i,b,kind='edited',model=None):
    rev=uid();path=WORK/'labels'/f'{rev}.json';atomic(path,b)
    write('INSERT INTO revisions VALUES(?,?,?,?,?,?)',(rev,i,kind,str(path),model,time.time()));return rev
def txt(b,w,h):
    return ''.join(f"{r['class_id']} {(r['x1']+r['x2'])/2/w:.9f} {(r['y1']+r['y2'])/2/h:.9f} {(r['x2']-r['x1'])/w:.9f} {(r['y2']-r['y1'])/h:.9f}\n" for r in b)
def eligible():return query("SELECT * FROM images WHERE current IS NOT NULL AND status IN ('editing','approved')")
def sync():
    root=ROOT/'_store/pending';old=json.loads((root/'manifest.json').read_text()) if (root/'manifest.json').exists() else {'samples':[]}
    registry=read_registry();samples=[]
    for r in eligible():
        e=registry['entries'].get(r['origin'])
        if e and e['revision']==r['current']:continue
        image=f"images/{r['id']}{Path(r['path']).suffix}"; label=f"labels/{r['id']}.txt"
        copy(r['path'],root/image);p=root/label;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(txt(boxes(r['current']),r['width'],r['height']))
        samples.append({**r,'image':image,'label':label,'revision':r['current']})
    atomic(root/'manifest.json',{'names':NAMES,'samples':samples})
    keep={r['id'] for r in samples}
    for r in old['samples']:
        if r['id'] not in keep:
            for k in ('image','label'):
                p=(root/r[k]).resolve()
                if p.is_relative_to(root.resolve()):p.unlink(missing_ok=True)
    publish_layout()
def publish_layout(seed=None):
    with LOCK:
        return _publish_layout(seed)

def _publish_layout(seed=None):
    samples=[]
    for r in eligible():
        p=WORK/'labels'/f"{r['current']}.txt"
        if not p.exists():p.write_text(txt(boxes(r['current']),r['width'],r['height']))
        samples.append({'id':r['id'],'origin':r['origin'],'sha256':r['digest'],'revision':r['current'],'image':r['path'],'label':str(p)})
    unlabeled=[{'id':r['id'],'image':r['path']} for r in query("SELECT * FROM images WHERE status IN ('unreviewed','retry')")]
    return dataset_layout.publish(ROOT,samples,unlabeled,ROOT/'_store/registry.json','labels',seed=seed)
def read_registry():
    p=ROOT/'_store/registry.json'
    return json.loads(p.read_text()) if p.exists() else {'id':uid(),'entries':{},'hash_groups':{}}
class Import(BaseModel):path:str;session:str=''
class Box(BaseModel):
    id:str
    class_id:int=Field(ge=0,lt=len(NAMES))
    x1:float;y1:float;x2:float;y2:float
class Save(BaseModel):boxes:list[Box];expected_revision:str|None=None
class Status(BaseModel):status:str
class Apply(BaseModel):revision:str
class Infer(BaseModel):
    model:str
    images:list[str]=Field(min_length=1)
    confidence:float=Field(default=.25,gt=0,le=1)
class Model(BaseModel):path:str;name:str=''
class Train(BaseModel):
    model:str='red_green_200ep';epochs:int=Field(default=20,ge=1,le=1000);batch_size:int=Field(default=4,ge=1,le=32)
    lr:float=Field(default=.0001,gt=0,le=.1);seed:int=31
@app.get('/api/state')
def state():
    jobs=query('SELECT * FROM jobs ORDER BY created DESC LIMIT 30')
    for j in jobs:
        for k in ('payload','result'):j[k]=json.loads(j[k]) if j[k] else None
        p=WORK/'runs'/j['id']/'metrics.jsonl';j['metrics']=[]
        if p.exists():
            for line in p.read_text().splitlines():
                try:j['metrics'].append(json.loads(line))
                except ValueError:pass
    images=query('SELECT * FROM images ORDER BY session,name')
    for image in images:
        image['prediction_box_count']=prediction_box_count(image['prediction']) if image['prediction'] else None
        image['label_class_counts']=list(revision_class_counts(image['current'])) if image['current'] else [0]*len(NAMES)
        image['label_box_count']=sum(image['label_class_counts']) if image['current'] else None
    models=query('SELECT * FROM models')
    default_model=next((name for name in ('best','red_green_200ep') if any(m['id']==name for m in models)),models[0]['id'] if models else None)
    return {'images':images,'models':models,'default_model':default_model,'jobs':jobs,'classes':NAMES,'default_folder':str(ROOT/'unlabeled')}
@app.post('/api/import')
def import_images(body:Import):
    src=Path(body.path).expanduser().resolve()
    if not src.is_dir() or src==WORK or WORK in src.parents:raise HTTPException(400,'외부 이미지 폴더를 선택하세요')
    added=0;duplicates=0;scanned=0;ids=[];errors=[]
    with LOCK:
        for p in sorted(src.rglob('*')):
            if p.suffix.lower() not in ('.jpg','.jpeg','.png','.bmp','.webp') or not p.is_file():continue
            if any(p.resolve().is_relative_to(d.resolve()) for d in (WORK,ROOT/'_store',ROOT/'deleted')):continue
            scanned+=1
            try:
                digest=hashlib.sha256(p.read_bytes()).hexdigest();m=re.match(r'(.*?)__frame_(\d+)',p.name)
                session=m[1] if m else body.session or src.name;origin=f'{session}:{int(m[2])}' if m else digest
                if query('SELECT id FROM images WHERE digest=? OR origin=?',(digest,origin),True):
                    duplicates+=1;continue
                with Image.open(p) as im:
                    im.load();w,h=im.size
                    if w*h>25000000:raise ValueError('Image too large')
                i=uid();dest=WORK/'images'/(i+p.suffix.lower());copy(p,dest)
                write('INSERT INTO images VALUES(?,?,?,?,?,?,?,?,?,?,?)',(i,p.name,str(dest),digest,origin,session,w,h,'unreviewed',None,None));added+=1;ids.append(i)
            except Exception as e:errors.append({'file':p.name,'error':str(e)})
    dataset_layout.archive_imports(src,WORK,ROOT,query('SELECT * FROM images'))
    publish_layout()
    imported=[item(i) for i in ids]
    dataset=dataset_layout.record_import(ROOT,src,imported,duplicates,errors)
    return {'added':added,'duplicates':duplicates,'scanned':scanned,'ids':ids,'errors':errors,'dataset':dataset}
@app.get('/api/images/{i}/file')
def image(i:str):return FileResponse(item(i)['path'])
@app.get('/api/images/{i}/revisions')
def revisions(i:str):item(i);return query('SELECT * FROM revisions WHERE image_id=? ORDER BY created DESC',(i,))
@app.get('/api/revisions/{rev}/boxes')
def get_boxes(rev:str):return boxes(rev)
@app.post('/api/images/{i}/boxes')
def save(i:str,body:Save):
    with LOCK:
        r=item(i);editable(r)
        if body.expected_revision!=r['current']:raise HTTPException(409,'라벨이 변경되었습니다. 다시 불러오세요')
        b=[v.model_dump() for v in body.boxes]
        if len({v['id'] for v in b})!=len(b):raise HTTPException(400,'Duplicate box ID')
        for v in b:
            if not all(math.isfinite(v[k]) for k in ('x1','y1','x2','y2')) or not (0<=v['x1']<v['x2']<=r['width'] and 0<=v['y1']<v['y2']<=r['height']):raise HTTPException(400,'박스 좌표가 이미지 범위를 벗어났습니다')
        rev=revision(i,b);write("UPDATE images SET current=?,status='editing' WHERE id=?",(rev,i));sync()
    return {'revision':rev}
@app.post('/api/images/{i}/status')
def status(i:str,body:Status):
    with LOCK:
        r=item(i);editable(r)
        if body.status not in ('approved','retry','excluded','unreviewed'):raise HTTPException(400,'Unknown status')
        if body.status=='approved' and not r['current']:raise HTTPException(400,'먼저 라벨을 저장하세요')
        write('UPDATE images SET status=? WHERE id=?',(body.status,i));sync()
    return {'ok':True}
@app.post('/api/images/{i}/apply')
def apply(i:str,body:Apply):
    with LOCK:
        editable(item(i));r=query('SELECT * FROM revisions WHERE id=? AND image_id=?',(body.revision,i),True)
        if not r:raise HTTPException(404,'Revision not found')
        rev=revision(i,boxes(body.revision));write("UPDATE images SET current=?,status='editing' WHERE id=?",(rev,i));sync()
    return {'revision':rev}
@app.post('/api/images/{i}/delete')
def delete(i:str):
    with LOCK:
        r=item(i);root=ROOT/'deleted';copy(r['path'],root/'images'/Path(r['path']).name)
        revs=revisions(i)
        for rev in revs:copy(rev['path'],root/'history'/i/Path(rev['path']).name)
        if r['current']:
            p=root/'labels'/f'{i}.txt';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(txt(boxes(r['current']),r['width'],r['height']))
        atomic(root/'metadata'/f'{i}.json',{**r,'revisions':revs});write("UPDATE images SET status='deleted' WHERE id=?",(i,));sync()
    return {'ok':True}
def model(i):
    r=query('SELECT * FROM models WHERE id=?',(i,),True)
    if not r:raise HTTPException(404,'Model not found')
    return r
@app.post('/api/models')
def register(body:Model):
    p=Path(body.path).expanduser().resolve()
    if not p.is_file() or p.suffix!='.pt':raise HTTPException(400,'Existing .pt required')
    try:load(p)
    except Exception as e:raise HTTPException(400,str(e))
    r=query('SELECT * FROM models WHERE path=?',(str(p),),True)
    if r:return r
    i=uid();write('INSERT INTO models VALUES(?,?,?)',(i,body.name or p.stem,str(p)));return model(i)
def submit(kind,payload):
    i=uid();write('INSERT INTO jobs VALUES(?,?,?,?,?,?,?)',(i,kind,'queued',json.dumps(payload),None,None,time.time()));POOL.submit(run,i,kind,payload);return {'job':i}
@app.post('/api/infer')
def infer(body:Infer):
    model(body.model)
    for i in body.images:editable(item(i))
    return submit('infer',body.model_dump())
@app.post('/api/datasets')
def snapshot(seed:int=31):
    with LOCK:
        rows=eligible()
        if len(rows)<5:raise HTTPException(400,'저장·승인 이미지가 최소 5장 필요합니다')
        registry=read_registry();entries=registry['entries'];groups=registry['hash_groups']
        unseen=sorted([r for r in rows if r['origin'] not in entries],key=lambda r:hashlib.sha256(('31:'+r['origin']).encode()).hexdigest())
        need=max(0,round((len(entries)+len(unseen))*.2)-sum(e['group']=='eval' for e in entries.values()))
        for r in unseen:
            group=groups.get(r['digest'])
            if group is None:group='eval' if need>0 else 'pool';need-=int(group=='eval')
            entries[r['origin']]={'group':group};groups[r['digest']]=group
        for r in rows:
            e=entries[r['origin']];e.update(revision=r['current'],sha256=r['digest'])
            base=ROOT/'_store'/e['group'];copy(r['path'],base/'images'/Path(r['path']).name)
            p=base/'labels'/f"{r['id']}.txt";p.parent.mkdir(parents=True,exist_ok=True);p.write_text(txt(boxes(r['current']),r['width'],r['height']))
        atomic(ROOT/'_store/registry.json',registry);sync()
        pool=sorted([r for r in rows if entries[r['origin']]['group']=='pool'],key=lambda r:r['origin']);evaluation=sorted([r for r in rows if entries[r['origin']]['group']=='eval'],key=lambda r:r['origin'])
        if len(pool)<2 or not evaluation:raise HTTPException(400,'train/val 및 eval 이미지가 필요합니다')
        assignments=dataset_layout.split_assignments(rows,entries,seed)
        dest=ROOT/'_store/versions'/uid();samples=[];seen=set()
        for split,part in [('train',[r for r in pool if assignments[r['id']]=='train']),('val',[r for r in pool if assignments[r['id']]=='test']),('test',evaluation)]:
            for r in part:
                digest=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()
                if digest!=r['digest'] or digest in seen:raise HTTPException(409,'변경되었거나 중복된 이미지')
                seen.add(digest);copy(r['path'],dest/{'train':'train','val':'test','test':'eval'}[split]/'images'/Path(r['path']).name)
                disk_split={'train':'train','val':'test','test':'eval'}[split]
                p=dest/disk_split/'labels'/f"{r['id']}.txt";p.parent.mkdir(parents=True,exist_ok=True);p.write_text(txt(boxes(r['current']),r['width'],r['height']))
                samples.append({**r,'split':split,'group':entries[r['origin']]['group']})
        publish_layout(seed)
        # JSON is also valid YAML; all class IDs follow CSV order.
        data={'path':str(dest),'train':'train/images','val':'test/images','test':'eval/images','names':NAMES,'nc':len(NAMES)}
        atomic(dest/'dataset.yaml',data);atomic(dest/'manifest.json',{'policy_id':registry['id'],'samples':samples})
        return {'path':str(dest),'data':str(dest/'dataset.yaml'),'counts':{s:sum(r['split']==s for r in samples) for s in ('train','val','test')}}
@app.post('/api/train')
def train(body:Train):
    m=model(body.model);data=snapshot(body.seed)
    return submit('train',{**body.model_dump(),'weights':m['path'],'dataset':data})
def run(i,kind,payload):
    write("UPDATE jobs SET state='running' WHERE id=?",(i,))
    try:
        if kind=='infer':
            m=load(model(payload['model'])['path']);done=0;errors=[]
            for image_id in payload['images']:
                try:
                    r=item(image_id)
                    if r['status']=='deleted':continue
                    result=m.predict(r['path'],imgsz=640,conf=payload.get('confidence',.25),verbose=False)[0];b=[]
                    for xy,c,conf in zip(result.boxes.xyxy.tolist(),result.boxes.cls.tolist(),result.boxes.conf.tolist()):
                        x1,y1,x2,y2=xy;x1=max(0,min(r['width'],x1));x2=max(0,min(r['width'],x2));y1=max(0,min(r['height'],y1));y2=max(0,min(r['height'],y2))
                        if x2>x1 and y2>y1:b.append({'id':uid(),'class_id':NAMES.index(m.names[int(c)]),'x1':x1,'y1':y1,'x2':x2,'y2':y2})
                    with LOCK:
                        if item(image_id)['status']=='deleted':continue
                        rev=revision(image_id,b,'prediction',payload['model']);write("UPDATE images SET prediction=?,current=COALESCE(current,?),status=CASE WHEN status='retry' THEN 'unreviewed' ELSE status END WHERE id=?",(rev,rev,image_id))
                    done+=1
                except Exception as e:errors.append({'image':image_id,'error':str(e)})
                result={'completed':done,'errors':errors};write('UPDATE jobs SET result=? WHERE id=?',(json.dumps(result),i))
            result={'completed':done,'errors':errors}
        else:
            out=WORK/'runs'/i;out.mkdir(parents=True);config={**payload,'output':str(out),'data':payload['dataset']['data']};atomic(out/'request.json',config)
            log=WORK/'logs'/f'{i}.log'
            with log.open('w') as f:p=subprocess.run([sys.executable,str(HERE/'engine.py'),str(out/'request.json')],stdout=f,stderr=subprocess.STDOUT)
            if p.returncode:raise RuntimeError(log.read_text()[-3000:])
            result=json.loads((out/'result.json').read_text());mid=uid();write('INSERT INTO models VALUES(?,?,?)',(mid,'추가 학습 '+i[:6],result['best']));result['model']=model(mid)
        write("UPDATE jobs SET state='complete',result=? WHERE id=?",(json.dumps(result),i))
    except Exception as e:write("UPDATE jobs SET state='failed',error=? WHERE id=?",(str(e),i))
app.mount('/',StaticFiles(directory=HERE/'static',html=True),name='bbox-ui')
