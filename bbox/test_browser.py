"""Real browser check against an isolated bbox workspace (never user data)."""
import json,os,socket,subprocess,sys,tempfile,time,urllib.request
from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
def main():
 with tempfile.TemporaryDirectory(prefix='bbox-browser-') as temp:
  temp=Path(temp);source=temp/'input';source.mkdir();Image.new('RGB',(640,480),(30,40,55)).save(source/'sample.png')
  sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
  runner=temp/'serve.py';runner.write_text(f"import sys\nsys.path.insert(0,{str(ROOT)!r})\nfrom bbox import app as b\nfrom fastapi import FastAPI\nimport uvicorn\nb.initialize()\napp=FastAPI()\napp.mount('/bbox',b.app)\nuvicorn.run(app,host='127.0.0.1',port={port})\n")
  env={**os.environ,'BBOX_REVIEW_WORK':str(temp/'data/_store/review'),'BBOX_REVIEW_DATA':str(temp/'data'),'OMP_NUM_THREADS':'4'}
  with (temp/'server.log').open('w') as log:
   proc=subprocess.Popen([sys.executable,str(runner)],env=env,stdout=log,stderr=subprocess.STDOUT)
   try:
    for _ in range(100):
     try:urllib.request.urlopen(f'http://127.0.0.1:{port}/bbox/api/state',timeout=1);break
     except Exception:
      if proc.poll() is not None:raise RuntimeError((temp/'server.log').read_text())
      time.sleep(.1)
    with sync_playwright() as p:
     browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
     page=browser.new_page(viewport={'width':1600,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
     page.goto(f'http://127.0.0.1:{port}/bbox/');page.locator('#import').click();page.locator('#folder').fill(str(source));page.locator('#importForm button.primary').click();page.wait_for_function('picture!==null')
     def pos(x,y):return page.evaluate('([x,y])=>{let r=canvas.getBoundingClientRect();return [r.left+tx+x*scale,r.top+ty+y*scale]}',[x,y])
     def drag(a,b):
      page.mouse.move(*pos(*a));page.mouse.down();page.mouse.move(*pos(*b),steps=8);page.mouse.up()
     page.locator('#drawTool').click();drag((80,70),(250,200));assert page.evaluate('boxes.length')==1
     page.locator('#selectTool').click();drag((160,130),(180,150));assert abs(page.evaluate('boxes[0].x1')-100)<2
     edge=page.evaluate('[boxes[0].x2,boxes[0].y2]');drag(edge,(310,260));assert abs(page.evaluate('boxes[0].x2')-310)<2
     page.locator('#class').select_option('3');assert page.evaluate('boxes[0].class_id')==3
     page.locator('#undo').click();assert page.evaluate('boxes[0].class_id')==0
     page.locator('#redo').click();assert page.evaluate('boxes[0].class_id')==3
     page.locator('[data-box]').first.click();page.locator('#canvas').focus();page.keyboard.press('7');assert page.evaluate('boxes[0].class_id')==6
     page.keyboard.press('8');assert page.evaluate('boxes[0].class_id')==7
     assert page.evaluate('colors[6]')=='#ff4d5e' and page.evaluate('colors[7]')=='#39d979'
     page.locator('#class').select_option('3')
     page.locator('#save').click();page.wait_for_function('!dirty && !busy')
     manifest=json.loads((temp/'data/manifest.json').read_text());row=manifest['samples'][0]
     assert (temp/'data'/row['image']).stem==(temp/'data'/row['label']).stem
     assert (temp/'data'/row['label']).read_text().startswith('3 ')
     page.reload();page.locator('#filter').select_option('all');page.wait_for_function('current!==null');assert page.evaluate('boxes[0].class_id')==3
     page.screenshot(path='/tmp/annotation-bbox-editor.png')
     page.locator('#deleteImage').click();page.wait_for_function('current===null && !busy')
     assert len(list((temp/'data/deleted/labels').glob('*.txt')))==1
     assert len(list((temp/'data/deleted/history').rglob('*.json')))==1
     assert not errors,errors
     browser.close();print('Browser import, draw, move, resize, class, undo/redo, save/reload and deletion passed')
   finally:
    proc.terminate()
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
if __name__=='__main__':main()
