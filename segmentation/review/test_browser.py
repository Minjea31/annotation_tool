from playwright.sync_api import sync_playwright
import json, time, os, socket, subprocess, sys, tempfile, urllib.request
from pathlib import Path
def check_space_pan(page):
 # A held key emits repeats; every one must cancel the browser's page scroll.
 page.evaluate('document.activeElement.blur();window.scrollTo(0,0)')
 before=page.evaluate('({x:tx,y:ty,dirty})')
 box=page.locator('#canvas').bounding_box()
 x,y=box['x']+box['width']/2,box['y']+box['height']/2
 page.keyboard.down('Space')
 for _ in range(5):page.keyboard.down('Space')
 page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+60,y+30,steps=5)
 page.wait_for_timeout(400)
 assert page.evaluate('scrollY')==0, 'Held Space scrolled the page'
 after=page.evaluate('({x:tx,y:ty,dirty})')
 assert abs(after['x']-before['x']-60)<1 and abs(after['y']-before['y']-30)<1
 assert before['dirty']==after['dirty'], 'Panning edited the mask'
 page.mouse.up();page.keyboard.up('Space')
 assert not page.evaluate('space')
 # Clicking the canvas takes focus away from sliders/fields without scrolling.
 page.locator('#brush').focus()
 page.mouse.click(x,y,button='middle')
 assert page.evaluate('document.activeElement===canvas')
 page.locator('#search').focus();page.keyboard.press('Space')
 assert page.input_value('#search')==' ', 'Space blocked text input'
 page.locator('#search').fill('');page.locator('#search').blur()
 page.evaluate('fitImage();window.scrollTo(0,0)')

def check_all_infer_button(page):
 # Intercept only this test's queue submission; never launch a large real job.
 original_id=page.evaluate('current.id')
 fake=page.evaluate('JSON.parse(JSON.stringify(state))')
 fake['jobs']=[]
 for row in fake['images']:row['status']='approved'
 for row in fake['images'][:2]:row['status']='unreviewed'
 fake['images'][2]['status']='excluded'
 fake['images'][3]['status']='editing'
 expected={row['id'] for row in fake['images'][:2]}
 captured=[]
 def state_route(route):route.fulfill(json=fake)
 def infer_route(route):
  captured.append(route.request.post_data_json)
  route.fulfill(json={'job':'browser-test-intercept'})
 page.route('**/api/state',state_route)
 page.route('**/api/infer',infer_route)
 try:
  page.evaluate('refresh()')
  page.select_option('#filter','approved')
  page.locator('#search').fill('no-image-matches-this')
  # Hidden checkbox selections must not restrict or expand the global action.
  page.evaluate("selected=new Set([state.images[4].id])")
  page.wait_for_function("document.querySelector('#allInfer').textContent.includes('2장')")
  model=page.locator('#model').input_value()
  page.locator('#allInfer').click()
  page.wait_for_function('allInferSubmitting===false')
  assert len(captured)==1
  assert set(captured[0]['images'])==expected
  assert captured[0]['model']==model
  page.evaluate("state.jobs=[{kind:'infer',state:'running'}];renderAllInfer()")
  assert page.locator('#allInfer').is_disabled()
 finally:
  page.unroute('**/api/state',state_route);page.unroute('**/api/infer',infer_route)
  page.locator('#search').fill('')
  page.select_option('#filter','all')
  page.evaluate('refresh()')
  page.evaluate('(id)=>loadImage(id)',original_id)
  page.locator('#canvas').focus()

def exercise(base_url):
 with sync_playwright() as p:
  browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
  page=browser.new_page(viewport={'width':1560,'height':1050})
  errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto(base_url);page.wait_for_function('state.images.length>0');page.select_option('#filter','all');page.wait_for_function('image !== null')
  assert page.locator('[data-class]').count()==2
  assert page.locator('.jobs-panel').bounding_box()['y']>=page.viewport_size['height']
  assert page.locator('#viewport').bounding_box()['height']>page.viewport_size['height']*.6
  footer=page.locator('.footer').bounding_box()
  assert footer['y']+footer['height']<=page.viewport_size['height']
  check_space_pan(page)
  check_all_infer_button(page)
  original=page.evaluate('current.id');original_rev=page.evaluate('activeRevision')
  # Two class hotkeys and image-space brush coordinates at arbitrary zoom.
  page.keyboard.press('Digit2');assert page.evaluate('cls')==2
  page.keyboard.press('KeyA')
  pos=page.evaluate('(()=>{const r=canvas.getBoundingClientRect();return {x:r.left+tx+100*scale,y:r.top+ty+100*scale}})()')
  page.mouse.move(pos['x'],pos['y']);page.mouse.down();page.mouse.move(pos['x']+25,pos['y']+12,steps=6);page.mouse.up()
  assert page.evaluate('mask[100*current.width+100]')==2
  page.keyboard.press('KeyZ');page.keyboard.press('KeyX');assert page.evaluate('mask[100*current.width+100]')==2
  page.keyboard.press('KeyV');page.wait_for_function('dirty===false && activeRevision!=='+json.dumps(original_rev))
  edited=page.evaluate('activeRevision')
  # Persistence after browser reload.
  page.reload();page.wait_for_function('state.images.length>0');page.select_option('#filter','all');page.wait_for_function('image!==null');assert page.evaluate('mask[100*current.width+100]')==2
  # Polygon paint and erase, including closed polygon keyboard action.
  page.keyboard.press('Digit1');page.keyboard.press('KeyD')
  for x,y in [(180,130),(230,130),(205,180)]:
   pos=page.evaluate(f'(()=>{{const r=canvas.getBoundingClientRect();return {{x:r.left+tx+{x}*scale,y:r.top+ty+{y}*scale}}}})()');page.mouse.click(pos['x'],pos['y'])
  page.keyboard.press('KeyF');assert page.evaluate('mask[145*current.width+205]')==1
  page.keyboard.press('KeyS')
  pos=page.evaluate('(()=>{const r=canvas.getBoundingClientRect();return {x:r.left+tx+205*scale,y:r.top+ty+145*scale}})()');page.mouse.click(pos['x'],pos['y']);assert page.evaluate('mask[145*current.width+205]')==0
  # Input focus blocks shortcuts.
  page.locator('#search').fill('');page.locator('#search').press('r');assert page.evaluate('current.id')==original
  page.locator('#search').fill('');page.locator('#search').blur()
  page.keyboard.press('KeyR');page.wait_for_function('current.id!=='+json.dumps(original))
  page.select_option('#filter','retry');page.wait_for_timeout(200)
  assert page.locator('.image-row').count()==1
  page.locator('.image-row .name').click();page.wait_for_function('current.id==='+json.dumps(original))
  page.click('#batchInfer');page.wait_for_function("state.jobs.some(j=>j.kind==='infer'&&j.state==='complete')",timeout=30000)
  page.wait_for_timeout(500)
  assert page.evaluate('current.current!==current.prediction')
  # Paint directly on the displayed prediction and save a fork without altering it.
  assert page.locator('#viewPrediction').count()==0
  assert page.locator('#viewCurrent').inner_text()=='모델 예측'
  page.click('#viewCurrent');page.wait_for_function("view==='current'")
  prediction=page.evaluate('current.prediction')
  page.keyboard.press('KeyA');page.keyboard.press('Digit2')
  pos=page.evaluate('(()=>{const r=canvas.getBoundingClientRect();return {x:r.left+tx+100*scale,y:r.top+ty+100*scale}})()')
  page.mouse.click(pos['x'],pos['y'])
  assert page.evaluate('mask[100*current.width+100]')==2
  assert page.evaluate('view')=='current'
  page.keyboard.press('KeyV');page.wait_for_function('dirty===false')
  assert page.evaluate('current.prediction')==prediction
  assert page.evaluate('activeRevision')!=prediction
  page.select_option('#filter','all');page.click('#approveBtn');page.wait_for_timeout(500)
  # UI starts a real one-epoch GPU training job.
  page.click('#trainBtn');page.locator('#epochs').fill('1');page.locator('#batchSize').fill('4');page.locator('#trainForm button[type=submit]').click()
  print('Browser: waiting for training',flush=True)
  page.wait_for_function("state.jobs.some(j=>j.kind==='train'&&j.state==='complete')",timeout=120000)
  print('Browser: training completed',flush=True)
  assert page.evaluate("state.jobs.find(j=>j.kind==='train').metrics.length") == 1
  print('Browser: capturing render',flush=True)
  page.screenshot(path='/tmp/lane-review-tested.png',full_page=True,timeout=15000)
  print('Browser: render captured',flush=True)
  # Every pool image is editable; train/val/test belong only to snapshots.
  assert page.evaluate("state.images.every(i=>i.split==='train')")
  result=page.evaluate("state.jobs.find(j=>j.kind==='train').result")
  assert sum(result['dataset']['counts'].values())==169
  assert result['dataset']['counts']=={'train':101,'val':34,'test':34}
  assert 'test' in result
  # Delete sits immediately to the left of Save and archives the data with labels.
  assert page.locator('#deleteBtn').evaluate("e=>e.nextElementSibling.id")=='saveBtn'
  deleted_id=page.evaluate('current.id')
  print('Browser: testing archive delete',flush=True)
  page.click('#deleteBtn');page.wait_for_function('(id)=>state.images.find(i=>i.id===id).status==="deleted"',arg=deleted_id)
  assert page.locator('.image-row[data-id="'+deleted_id+'"]').count()==0
  delete_root=page.evaluate('state.delete_folder')
  assert Path(delete_root,'metadata',deleted_id+'.json').is_file()
  archived=json.loads(Path(delete_root,'metadata',deleted_id+'.json').read_text())
  assert Path(delete_root,archived['image']).is_file() and Path(delete_root,archived['mask']).is_file()
  page.select_option('#filter','deleted');page.wait_for_function("current&&current.status==='deleted'")
  assert page.locator('.image-row').count()==1
  assert not errors,errors
  print(json.dumps({'browser_errors':errors,'checks':['two classes','brush coordinates','undo/redo','save/reload','polygon','eraser','input focus','retry shortcut','model inference','preserve edits','direct prediction editing','training via UI','6:2:2 snapshot and test evaluation'],'screenshot':'/tmp/lane-review-tested.png'},ensure_ascii=False))
  browser.close()

if __name__ == '__main__':
 with tempfile.TemporaryDirectory(prefix='lane-review-browser-') as temp:
  with socket.socket() as sock:
   sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
  with open(Path(temp)/'server.log','w') as log:
   process=subprocess.Popen([sys.executable,str(Path(__file__).with_name('app.py'))],
    env={**os.environ,'LANE_REVIEW_WORK':temp+'/work','LANE_REVIEW_DELETE':temp+'/delete_dataset','LANE_REVIEW_APPROVE':temp+'/approve_dataset','LANE_REVIEW_PORT':str(port)},stdout=log,stderr=log)
   try:
    base_url=f'http://127.0.0.1:{port}'
    for _ in range(120):
     try:
      urllib.request.urlopen(base_url+'/api/state',timeout=1);break
     except Exception:
      if process.poll() is not None:raise RuntimeError('Server failed; '+Path(temp,'server.log').read_text())
      time.sleep(.25)
    exercise(base_url)
   finally:
    process.terminate();process.wait(timeout=15)
