"""One local origin, two isolated annotation workspaces, one GPU queue."""
import importlib.util, os, sys
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('ANNOTATION_SEG_DATA',str(ROOT/'segmentation/dataset'))
os.environ.setdefault('BBOX_REVIEW_DATA',str(ROOT/'bbox/dataset'))
sys.path.insert(0,str(ROOT/'segmentation/review'))
spec=importlib.util.spec_from_file_location('segmentation_app',ROOT/'segmentation/review/app.py')
seg=importlib.util.module_from_spec(spec);spec.loader.exec_module(seg)
from bbox import app as bbox
pool=ThreadPoolExecutor(max_workers=1);seg.POOL.shutdown();bbox.POOL.shutdown();seg.POOL=bbox.POOL=pool
@asynccontextmanager
async def lifespan(app):
    seg.initialize();bbox.initialize()
    yield
    pool.shutdown(wait=True)
app=FastAPI(lifespan=lifespan)
@app.middleware('http')
async def origin(request:Request,call_next):
    o=request.headers.get('origin')
    if o and o.rstrip('/')!=str(request.base_url).rstrip('/'):return JSONResponse({'detail':'Local same-origin access only'},status_code=403)
    return await call_next(request)
@app.get('/',response_class=HTMLResponse)
def home():return (ROOT/'launcher.html').read_text()
app.mount('/segmentation',seg.app)
app.mount('/bbox',bbox.app)
if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=int(os.environ.get('ANNOTATION_PORT','8765')))
