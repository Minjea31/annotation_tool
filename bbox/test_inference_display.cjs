// Exercise the real UI script with API responses and a recording canvas.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');

async function editor({predicted=false,edited=false,insecure=false,stateFailures=0}={}){
 const box={id:'prediction-box',class_id:2,x1:30,y1:40,x2:130,y2:140};
 const editedBox={...box,id:'edited-box',class_id:3,x1:200,x2:300};
 const row={id:'image',name:'sample.png',session:'bag',status:'unreviewed',current:predicted?'prediction':null,prediction:predicted?'prediction':null};
 const data={images:[row],models:[{id:'baseline',name:'baseline'}],jobs:[],classes:['slow','bump','crosswalk','left','right','parking'],default_folder:'/input'};
 const revisions=predicted?[{id:'prediction',kind:'prediction',created:1}]:[];
 const labels={prediction:[box],edited:[editedBox]};
 if(edited){row.current='edited';revisions.push({id:'edited',kind:'edited',created:2});}
 const strokes=[],texts=[],requests=[],elements=new Map(),listeners={};
 let pollCallback,stateGate,releaseState;
 const context={clearRect(){strokes.length=0;texts.length=0;},save(){},restore(){},translate(){},scale(){},drawImage(){},fillRect(){},strokeRect(...args){strokes.push(args);},fillText(s){texts.push(s);},measureText(s){return {width:s.length*7};}};
 function element(id){
  if(!elements.has(id))elements.set(id,{value:id==='filter'?'pending':id==='confidence'?'0.25':id==='model'?'baseline':'',options:[],dataset:{},style:{},clientWidth:900,clientHeight:700,
   textContent:'',innerHTML:'',classList:{toggle(){}},focus(){},setPointerCapture(){},click(){return this.onclick?.();},setAttribute(k,v){this[k]=v;},getContext(){return context;},getBoundingClientRect(){return {left:0,top:0};}});
  return elements.get(id);
 }
 const sandbox={console,structuredClone,setTimeout,clearTimeout,setInterval(callback){pollCallback=callback;},crypto:insecure?{getRandomValues:array=>require('node:crypto').webcrypto.getRandomValues(array)}:require('node:crypto'),ResizeObserver:class{observe(){}},
  Image:class{width=640;height=480;async decode(){}},window:{},
  document:{querySelector(s){return s==='dialog[open]'?null:element(s.slice(1));},querySelectorAll(){return [];},addEventListener(type,callback){listeners[type]=callback;}},
  async fetch(url,options={}){
   requests.push({url,...options});
   let value;
   if(url==='api/state'){
    if(stateFailures>0){stateFailures--;throw Error('Temporary connection failure');}
    if(stateGate)await stateGate;
    value=data;
   }
   else if(url==='api/infer')value={job:'queued'};
   else if(/^api\/images\/[^/]+\/revisions$/.test(url))value=revisions;
   else if(url.startsWith('api/revisions/'))value=labels[url.split('/')[2]];
   else throw Error('Unexpected request: '+url);
   return {ok:true,json:async()=>structuredClone(value)};
  }};
 vm.createContext(sandbox);
 vm.runInContext(fs.readFileSync(path.join(__dirname,'static/app.js'),'utf8'),sandbox);
 await new Promise(resolve=>setImmediate(resolve));
 return {data,row,revisions,labels,box,strokes,texts,requests,element,
  poll:()=>pollCallback(),
  holdState(){stateGate=new Promise(resolve=>{releaseState=resolve;});},
  releaseState(){releaseState();stateGate=null;},
  run:code=>vm.runInContext(code,sandbox),
  pointer(type,x,y){const e=vm.runInContext(`({button:0,pointerId:1,clientX:tx+${x}*scale,clientY:ty+${y}*scale,preventDefault(){}})`,sandbox);element('canvas')['onpointer'+type](e);},
  key(code){listeners.keydown({code,repeat:false,target:{matches(){return false;}},preventDefault(){}});},
  predict(id='prediction',boxes=[box]){row.prediction=id;row.current??=id;labels[id]=boxes;revisions.push({id,kind:'prediction',created:3});}};
}

test('existing inference draws bbox coordinates and class name on opening',async()=>{
 const ui=await editor({predicted:true});
 assert.deepEqual(ui.strokes,[[30,40,100,100]]);
 assert.deepEqual(ui.texts,['crosswalk']);
 assert.equal(ui.element('predictionInfo').textContent,'모델 예측 · 1개');
});

test('initial connection failure retries both the library and the image',async()=>{
 const ui=await editor({predicted:true,stateFailures:1});
 assert.equal(ui.run('current'),null);
 assert.match(ui.element('count').textContent,/연결 확인/);
 await ui.poll();
 assert.equal(ui.run('current.id'),'image');
 assert.equal(ui.run('picture!==null'),true);
 assert.deepEqual(ui.strokes,[[30,40,100,100]]);
 assert.equal(ui.element('count').textContent,'1 / 1');
});

test('a slow library request is not duplicated by the next poll',async()=>{
 const ui=await editor({predicted:true});
 const before=ui.requests.filter(r=>r.url==='api/state').length;
 ui.holdState();
 const first=ui.poll(),second=ui.poll();
 await new Promise(resolve=>setImmediate(resolve));
 assert.equal(ui.requests.filter(r=>r.url==='api/state').length,before+1);
 ui.releaseState();await Promise.all([first,second]);
 assert.equal(ui.run('current.id'),'image');
});

test('individual prediction appears while the batch is still running',async()=>{
 const ui=await editor();
 assert.equal(ui.element('predictionInfo').textContent,'추론 전');
 ui.data.jobs=[{id:'batch',kind:'infer',state:'running',payload:{images:['image','other']}}];
 ui.predict();
 await ui.run('refresh()');
 assert.deepEqual(ui.strokes,[[30,40,100,100]]);
 assert.equal(ui.element('predictionInfo').textContent,'모델 예측 · 1개');
 assert.equal(ui.data.jobs[0].state,'running');
});

test('pending prediction waits for editing to finish and is not lost between polls',async()=>{
 const ui=await editor({predicted:true});
 ui.run('dirty=true;boxes[0].x1=45;render()');
 ui.predict('new-prediction',[{...ui.box,x1:60}]);
 await ui.run('refresh()');
 await ui.run('refresh()');
 assert.equal(ui.strokes[0][0],45);
 ui.run('dirty=false');
 await ui.run('refresh()');
 assert.equal(ui.strokes[0][0],60);
});

test('prediction refresh preserves history and active pointer gestures',async()=>{
 const ui=await editor({predicted:true});
 ui.run('history=true');ui.predict('new-prediction',[{...ui.box,x1:60}]);
 await ui.run('refresh()');assert.equal(ui.strokes[0][0],30);
 ui.run('history=false;gesture={type:"move"}');
 await ui.run('refresh()');assert.equal(ui.strokes[0][0],30);
 ui.run('gesture=null');await ui.run('refresh()');assert.equal(ui.strokes[0][0],60);
});

test('opening a predicted image restores bbox visibility after hiding boxes',async()=>{
 const ui=await editor({predicted:true});
 ui.element('original').onclick();assert.equal(ui.strokes.length,0);
 assert.equal(ui.element('original').textContent,'박스 표시');
 await ui.run('open("image")');
 assert.equal(ui.strokes.length,1);
 assert.equal(ui.element('original')['aria-pressed'],'true');
});

test('empty prediction is explicitly shown as no detections',async()=>{
 const ui=await editor();ui.predict('empty',[]);
 await ui.run('refresh()');
 assert.equal(ui.strokes.length,0);
 assert.equal(ui.element('predictionInfo').textContent,'모델 예측 · 검출 없음');
});

test('a newer saved edit remains the displayed label',async()=>{
 const ui=await editor({predicted:true,edited:true});
 assert.deepEqual(ui.strokes,[[200,40,100,100]]);
 assert.deepEqual(ui.texts,['left']);
 assert.equal(ui.element('predictionInfo').textContent,'저장된 라벨 · 1개');
});

test('a newer empty prediction cannot hide approved saved labels',async()=>{
 const ui=await editor({predicted:true,edited:true});
 ui.row.status='approved';ui.predict('empty-prediction',[]);
 await ui.run('refresh()');
 assert.deepEqual(ui.strokes,[[200,40,100,100]]);
 assert.equal(ui.element('predictionInfo').textContent,'승인된 라벨 · 1개');
 assert.equal(ui.run('boxes[0].id'),'edited-box');
});

test('newer model detections cannot replace manual edits in the editor',async()=>{
 const ui=await editor({predicted:true,edited:true});
 ui.row.status='editing';ui.predict('new-prediction',[{...ui.box,x1:60}]);
 await ui.run('refresh()');
 assert.deepEqual(ui.strokes,[[200,40,100,100]]);
 assert.equal(ui.element('predictionInfo').textContent,'저장된 라벨 · 1개');
});

test('approved current prediction is displayed instead of a newer inference',async()=>{
 const ui=await editor({predicted:true});
 ui.row.status='approved';ui.predict('new-prediction',[]);
 await ui.run('refresh()');
 assert.deepEqual(ui.strokes,[[30,40,100,100]]);
 assert.equal(ui.element('predictionInfo').textContent,'승인된 라벨 · 1개');
});

test('empty saved labels remain empty when inference later finds objects',async()=>{
 const ui=await editor({predicted:true,edited:true});ui.labels.edited=[];
 ui.row.status='approved';ui.predict('new-prediction',[ui.box]);
 await ui.run('refresh()');
 assert.equal(ui.strokes.length,0);
 assert.equal(ui.element('predictionInfo').textContent,'승인된 라벨 · 0개');
});

test('current, selected and all-image inference submit the chosen confidence',async()=>{
 const ui=await editor();ui.element('confidence').value='0.6';
 for(const id of ['infer','batchInfer','allInfer'])await ui.element(id).onclick();
 const submissions=ui.requests.filter(r=>r.url==='api/infer').map(r=>JSON.parse(r.body));
 assert.equal(submissions.length,3);
 for(const body of submissions){assert.equal(body.confidence,0.6);assert.deepEqual(body.images,['image']);}
});

test('invalid confidence does not queue inference',async()=>{
 const ui=await editor();
 for(const value of ['','0','-1','1.1','NaN']){
  ui.element('confidence').value=value;
  await assert.rejects(ui.run('infer(["image"])'),/Confidence/);
 }
 assert.equal(ui.requests.filter(r=>r.url==='api/infer').length,0);
});

test('detected filter lists images with predictions and excludes empty and deleted images',async()=>{
 const ui=await editor({predicted:true});ui.row.prediction_box_count=1;
 ui.data.images.push({...ui.row,id:'empty',prediction_box_count:0},{...ui.row,id:'new',prediction_box_count:null},{...ui.row,id:'deleted',status:'deleted',prediction_box_count:2});
 ui.element('filter').value='detected';
 await ui.run('refresh()');
 assert.deepEqual(Array.from(ui.run('filtered().map(r=>r.id)')),['image']);
 ui.run('list()');assert.match(ui.element('images').innerHTML,/예측 박스 1개/);
});

test('clicking an inferred box selects it without changing the annotation',async()=>{
 const ui=await editor({predicted:true});
 ui.pointer('down',80,90);ui.pointer('up',80,90);
 assert.equal(ui.run('selected'),'prediction-box');
 assert.equal(ui.run('dirty'),false);
 assert.equal(ui.run('undo.length'),0);
 assert.equal(ui.element('class').value,2);
 assert.match(ui.element('selectionInfo').textContent,/crosswalk/);
 assert.equal(ui.element('removeBox').disabled,false);
});

test('an existing box is selected even while the draw tool is active',async()=>{
 const ui=await editor({predicted:true});ui.element('drawTool').click();
 ui.pointer('down',80,90);ui.pointer('up',80,90);
 assert.equal(ui.run('selected'),'prediction-box');
 assert.equal(ui.run('boxes.length'),1);
 assert.equal(ui.run('tool'),'select');
 assert.equal(ui.run('dirty'),false);
});

test('box labels and the area near the visible border can be clicked',async()=>{
 const ui=await editor({predicted:true});
 const label=ui.run('labelBounds(boxes[0])');
 ui.pointer('down',label.x+label.w/2,label.y+label.h/2);ui.pointer('up',0,0);
 assert.equal(ui.run('selected'),'prediction-box');
 ui.run('selectBox(null)');ui.pointer('down',27,90);ui.pointer('up',27,90);
 assert.equal(ui.run('selected'),'prediction-box');
});

test('S removes the clicked model box and undo restores it',async()=>{
 const ui=await editor({predicted:true});
 ui.pointer('down',80,90);ui.pointer('up',80,90);ui.key('KeyS');
 assert.equal(ui.run('boxes.length'),0);assert.equal(ui.run('dirty'),true);
 assert.equal(ui.element('removeBox').disabled,true);
 ui.key('KeyZ');assert.equal(ui.run('boxes[0].id'),'prediction-box');
 assert.equal(ui.run('boxes[0].x1'),30);
});

test('clicking empty space clears the selection',async()=>{
 const ui=await editor({predicted:true});
 ui.pointer('down',80,90);ui.pointer('up',80,90);
 ui.pointer('down',400,350);ui.pointer('up',400,350);
 assert.equal(ui.run('selected'),null);assert.equal(ui.element('removeBox').disabled,true);
 assert.equal(ui.run('dirty'),false);
});

test('dragging still moves a selected bbox and creates one undo entry',async()=>{
 const ui=await editor({predicted:true});
 ui.pointer('down',80,90);ui.pointer('move',100,110);ui.pointer('up',100,110);
 assert.ok(Math.abs(ui.run('boxes[0].x1')-50)<1e-6);assert.ok(Math.abs(ui.run('boxes[0].y1')-60)<1e-6);
 assert.equal(ui.run('undo.length'),1);
 ui.key('KeyZ');assert.equal(ui.run('boxes[0].x1'),30);
});

test('drawing a new bbox still works in empty space',async()=>{
 const ui=await editor({predicted:true});ui.element('drawTool').click();
 ui.pointer('down',250,220);ui.pointer('move',350,300);ui.pointer('up',350,300);
 assert.equal(ui.run('boxes.length'),2);
 const coords=Array.from(ui.run('[boxes[1].x1,boxes[1].y1,boxes[1].x2,boxes[1].y2]'));
 [250,220,350,300].forEach((expected,i)=>assert.ok(Math.abs(coords[i]-expected)<1e-6));
 assert.equal(ui.run('dirty'),true);
});

test('D draws boxes over LAN HTTP without crypto.randomUUID',async()=>{
 const ui=await editor({predicted:true,insecure:true});
 assert.equal(ui.run('typeof crypto.randomUUID'),'undefined');
 ui.key('KeyD');assert.equal(ui.run('tool'),'draw');
 ui.pointer('down',250,220);ui.pointer('move',350,300);ui.pointer('up',350,300);
 ui.pointer('down',450,320);ui.pointer('move',550,400);ui.pointer('up',550,400);
 assert.equal(ui.run('boxes.length'),3);
 const ids=Array.from(ui.run('boxes.slice(1).map(b=>b.id)'));
 ids.forEach(id=>assert.match(id,/^[0-9a-f]{32}$/));
 assert.notEqual(ids[0],ids[1]);assert.equal(ui.run('dirty'),true);
 ui.key('KeyZ');assert.equal(ui.run('boxes.length'),2);
});

test('selected box corner handles still resize and can be undone',async()=>{
 const ui=await editor({predicted:true});
 ui.pointer('down',80,90);ui.pointer('up',80,90);
 ui.pointer('down',130,140);ui.pointer('move',170,180);ui.pointer('up',170,180);
 assert.ok(Math.abs(ui.run('boxes[0].x2')-170)<1e-6);
 assert.ok(Math.abs(ui.run('boxes[0].y2')-180)<1e-6);
 ui.key('KeyZ');assert.equal(ui.run('boxes[0].x2'),130);
});

test('running training displays only the latest mAP50 and mAP50-95 metrics',async()=>{
 const ui=await editor();
 ui.data.jobs=[{kind:'train',state:'running',metrics:[{epoch:1,metrics:{'metrics/mAP50(B)':.1,'metrics/mAP50-95(B)':.05}},{epoch:2,metrics:{'metrics/mAP50(B)':.8123,'metrics/mAP50-95(B)':.6543,'metrics/precision(B)':.9,'metrics/recall(B)':.8,fitness:.65}}]}];
 await ui.run('refresh()');const html=ui.element('jobs').innerHTML;
 assert.match(html,/검증 · mAP50 81\.23% · mAP50-95 65\.43%/);
 assert.doesNotMatch(html,/precision|recall|fitness|10\.00%/);
});

test('completed training shows final eval mAP instead of last epoch validation metrics',async()=>{
 const ui=await editor();
 ui.data.jobs=[{kind:'train',state:'complete',metrics:[{epoch:2,metrics:{'metrics/mAP50(B)':.99,'metrics/mAP50-95(B)':.88}}],result:{eval:{'metrics/mAP50(B)':.930111,'metrics/mAP50-95(B)':.908147,'metrics/precision(B)':.96,fitness:.908147}}}];
 await ui.run('refresh()');const html=ui.element('jobs').innerHTML;
 assert.match(html,/최종 eval · mAP50 93\.01% · mAP50-95 90\.81%/);
 assert.doesNotMatch(html,/99\.00%|88\.00%|precision|fitness/);
});

test('missing training metrics are shown as unavailable and zero is preserved',async()=>{
 const ui=await editor();
 ui.data.jobs=[{kind:'train',state:'running',metrics:[{epoch:1,metrics:{'metrics/mAP50(B)':0}}]}];
 await ui.run('refresh()');
 assert.match(ui.element('jobs').innerHTML,/mAP50 0\.00% · mAP50-95 —/);
});

async function approvedEditor(){
 const ui=await editor();
 ui.data.images=[
  {...ui.row,status:'approved',name:'z.png',label_box_count:3,label_class_counts:[2,0,1,0,0,0],prediction_box_count:99},
  {...ui.row,id:'small',status:'approved',name:'alpha.png',label_box_count:1,label_class_counts:[0,1,0,0,0,0]},
  {...ui.row,id:'empty',status:'approved',name:'empty.png',label_box_count:0,label_class_counts:[0,0,0,0,0,0]},
  {...ui.row,id:'pending',label_box_count:100,label_class_counts:[100,0,0,0,0,0]},
  {...ui.row,id:'deleted',status:'deleted',label_box_count:50,label_class_counts:[50,0,0,0,0,0]}
 ];
 ui.element('filter').value='approved';await ui.run('refresh()');return ui;
}

test('approved summary counts saved boxes and classes from approved images only',async()=>{
 const ui=await approvedEditor();
 assert.equal(ui.element('approvedControls').hidden,false);
 assert.equal(ui.element('approvedSummary').textContent,'승인 3장 · 라벨 4개');
 assert.match(ui.element('approvedClassStats').innerHTML,/slow<\/span><b>2개/);
 assert.match(ui.element('approvedClassStats').innerHTML,/bump<\/span><b>1개/);
 assert.match(ui.element('images').innerHTML,/라벨 3개 · slow 2, crosswalk 1/);
 assert.doesNotMatch(ui.element('images').innerHTML,/예측 박스 99개/);
});

test('approved list sorts by saved box counts, CSV classes and filename',async()=>{
 const ui=await approvedEditor();
 for(const [sort,ids] of [['count-desc',['image','small','empty']],['count-asc',['empty','small','image']],['class',['image','small','empty']],['name',['small','empty','image']]]){
  ui.element('approvedSort').value=sort;
  assert.deepEqual(Array.from(ui.run('filtered().map(r=>r.id)')),ids);
 }
});

test('approved class filter uses all classes in mixed labels and supports empty labels',async()=>{
 const ui=await approvedEditor();
 ui.element('approvedClass').value='2';
 assert.deepEqual(Array.from(ui.run('filtered().map(r=>r.id)')),['image']);
 await ui.run('refresh()');assert.equal(ui.element('approvedClass').value,'2');
 ui.element('approvedClass').value='empty';
 assert.deepEqual(Array.from(ui.run('filtered().map(r=>r.id)')),['empty']);
 ui.element('approvedClass').value='5';
 assert.equal(ui.run('filtered().length'),0);
});

test('approved search combines with class filtering and controls do not affect other statuses',async()=>{
 const ui=await approvedEditor();ui.element('approvedClass').value='2';ui.element('search').value='alpha';
 assert.equal(ui.run('filtered().length'),0);
 ui.element('filter').value='pending';ui.element('search').value='';ui.run('list()');
 assert.equal(ui.element('approvedControls').hidden,true);
 assert.deepEqual(Array.from(ui.run('filtered().map(r=>r.id)')),['pending']);
});

test('changing approved class opens a matching image for review',async()=>{
 const ui=await approvedEditor();ui.element('approvedClass').value='1';
 await ui.element('approvedClass').onchange();
 assert.equal(ui.run('current.id'),'small');
 assert.equal(ui.element('name').textContent,'alpha.png');
});
