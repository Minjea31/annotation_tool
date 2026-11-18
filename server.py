"""One shared server, two isolated annotation workspaces, one GPU queue."""
import errno, importlib.util, os, socket, sys
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.gzip import GZipMiddleware
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
app.add_middleware(GZipMiddleware,minimum_size=1024,compresslevel=3)
@app.middleware('http')
async def origin(request:Request,call_next):
    o=request.headers.get('origin')
    if o and o.rstrip('/')!=str(request.base_url).rstrip('/'):return JSONResponse({'detail':'Local same-origin access only'},status_code=403)
    return await call_next(request)
@app.get('/',response_class=HTMLResponse)
def home():return (ROOT/'launcher.html').read_text()
app.mount('/segmentation',seg.app)
app.mount('/bbox',bbox.app)
def print_connection_info(host,port):
    if host=='0.0.0.0':
        print(f'내 컴퓨터: http://127.0.0.1:{port}',flush=True)
        try:
            # Select the local address used by the default route; UDP sends no data.
            with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as connection:
                connection.connect(('10.255.255.255',1))
                address=connection.getsockname()[0]
            print(f'같은 Wi-Fi에서 접속: http://{address}:{port}',flush=True)
        except OSError:
            print(f'같은 Wi-Fi에서 접속: http://<이 컴퓨터의 Wi-Fi IPv4 주소>:{port}',flush=True)
    else:
        print(f'Annotation: http://{host}:{port}',flush=True)

if __name__=='__main__':
    import uvicorn
    host=os.environ.get('ANNOTATION_HOST','0.0.0.0')
    port=int(os.environ.get('ANNOTATION_PORT','8765'))
    try:
        # Reserve the port before initializing storage or changing job states.
        listener=socket.create_server((host,port),family=socket.AF_INET6 if ':' in host else socket.AF_INET)
    except OSError as error:
        if error.errno!=errno.EADDRINUSE:raise
        print(f'{port} 포트를 이미 사용 중입니다. 실행 중인 서버가 있다면 아래 주소로 접속하세요.',file=sys.stderr,flush=True)
        print_connection_info(host,port)
        print('다시 실행하려면 기존 서버 터미널에서 Ctrl+C로 종료한 뒤 ./run.sh를 실행하세요.',file=sys.stderr,flush=True)
        sys.exit(1)
    with listener:
        print_connection_info(host,port)
        uvicorn.run(app,host=host,port=port,fd=listener.fileno())
