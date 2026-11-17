'use strict';
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
let state={images:[],models:[],jobs:[]}, current=null, image=null, mask=null, activeRevision=null, shownRevision=null, editorRevision=null, dirty=false, view='current', tool='brush', cls=1, undo=[], redo=[], polygon=[], selected=new Set(), scale=1, tx=0,ty=0, space=false, dragging=false,last=null, pan=false, loadToken=0,busy=false, toastTimer, allInferSubmitting=false;
const canvas=$('#canvas'),ctx=canvas.getContext('2d'),overlay=document.createElement('canvas'),oc=overlay.getContext('2d');
const names={unreviewed:'미검수',editing:'수정 중',approved:'승인됨',retry:'재추론 대기',excluded:'제외',deleted:'삭제 보관'};
function escape(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function toast(s){$('#toast').textContent=s;$('#toast').style.display='block';clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').style.display='none',6000);}
async function api(path,body){const r=await fetch('api'+path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});if(!r.ok){let d=await r.json();throw Error(typeof d.detail==='string'?d.detail:JSON.stringify(d.detail));}return r.json();}
function safe(fn){return async(...a)=>{try{await fn(...a);}catch(e){toast(e.message);console.error(e);}};}
function filtered(){const f=$('#filter').value,q=$('#search').value.toLowerCase();return state.images.filter(i=>(i.status!=='deleted'||f==='deleted')&&(f==='all'||f==='pending'&&['unreviewed','retry'].includes(i.status)||f==='completed'&&['approved','editing'].includes(i.status)||i.status===f)&&(!q||(i.name+' '+i.session).toLowerCase().includes(q)));}
function renderList(){const rows=filtered(),visible=new Set(rows.map(i=>i.id));selected=new Set([...selected].filter(id=>visible.has(id)));$('#total').textContent=rows.length+' / '+state.images.length;$('#summary').textContent=`승인 ${state.images.filter(i=>i.status==='approved').length} · 재추론 ${state.images.filter(i=>i.status==='retry').length} · 수정 ${state.images.filter(i=>i.status==='editing').length}`;$('#selectedCount').textContent=selected.size?`${selected.size}개 선택`:'';$('#images').innerHTML=rows.map(i=>`<div class="image-row ${current?.id===i.id?'selected':''}" data-id="${i.id}"><input type="checkbox" ${selected.has(i.id)?'checked':''} aria-label="이미지 선택"><div><div class="name" title="${escape(i.name)}">${escape(i.name)}</div><small><span class="dot ${i.status}"></span>${names[i.status]} · ${escape(i.session)}</small></div></div>`).join('');$$('.image-row').forEach(el=>{el.onclick=safe(async e=>{if(e.target.type==='checkbox'){e.target.checked?selected.add(el.dataset.id):selected.delete(el.dataset.id);$('#selectedCount').textContent=`${selected.size}개 선택`;return;}await navigate(el.dataset.id);});});}
function modelOptions(){
 const old=$('#model').value;
 $('#model').innerHTML=state.models.map(m=>`<option value="${m.id}">${escape(m.name)}${m.legacy?' · 이전 분할':''}</option>`).join('');
 if(state.models.some(m=>m.id===old))$('#model').value=old;
 const choice=$('#trainModel').value;
 $('#trainModel').innerHTML='<option value="fresh">ImageNet 초기 가중치 · 새로 학습</option>'+state.models.filter(m=>m.train_eligible).map(m=>`<option value="${m.id}">${escape(m.name)}</option>`).join('');
 if(choice==='fresh'||state.models.some(m=>m.id===choice&&m.train_eligible))$('#trainModel').value=choice;
 const d=state.dataset;
 if(d)$('#datasetInfo').textContent=`승인 대기 ${d.pending}장 · train/test 풀 ${d.pool}장 · 고정 eval ${d.eval}장`;
}
function trainingDefaults(){const fresh=$('#trainModel').value==='fresh';$('#epochs').value=fresh?80:20;$('#lr').value=fresh?'0.001':'0.0001';}
$('#trainModel').onchange=trainingDefaults;

function renderJobs(){const labels={queued:'대기 중',running:'진행 중',complete:'완료',failed:'실패',interrupted:'서버 재시작으로 중단'};$('#jobs').innerHTML=state.jobs.length?state.jobs.slice(0,6).map(j=>{const m=j.metrics.at(-1), best=j.result?.best, mId=j.result?.model?.id;return `<div class="job"><div class="job-head"><b>${j.kind==='train'?'모델 추가 학습':'라벨 초안 추론'} · ${labels[j.state]}</b><span>${j.id.slice(0,6)}</span></div>${j.kind==='train'?`<p>640×480 · ${j.payload.epochs} epoch · ${j.payload.dataset?.policy_id?'고정 eval':'이전 분할'} · 데이터 ${j.payload.dataset?.counts?.train??''} train / ${j.payload.dataset?.counts?.val??''} test /  ${j.payload.dataset?.counts?.test??'—'} eval</p><progress value="${m?.epoch??0}" max="${j.payload.epochs}"></progress>${m?`<p>${m.epoch}/${j.payload.epochs} · loss ${m.train.loss.toFixed(3)} · 검증 IoU ${(m.val.lane_miou*100).toFixed(1)}% · 주황 ${(m.val.iou[1]*100).toFixed(1)}% / 점선 ${(m.val.iou[2]*100).toFixed(1)}%</p>`:''}${best?`<p>최고 epoch ${best.epoch} · 검증 IoU ${(best.val.lane_miou*100).toFixed(1)}% · 시작 모델 ${j.payload.model==='fresh'?'ImageNet 초기 가중치':escape(state.models.find(x=>x.id===j.payload.model)?.name)}</p>`:''}`:`<p>${j.result?.completed??0}/${j.payload.images.length}장 · 실패 ${j.result?.errors?.length??0}</p>`}${j.error?`<p class="error">${escape(j.error)}</p>`:''}${j.result?.errors?.length?`<p class="error">${escape(j.result.errors.map(e=>e.error).join('\n'))}</p>`:''}${j.result?.test?`<p>최종 eval IoU ${(j.result.test.lane_miou*100).toFixed(1)}% · 모델 선택에는 미사용</p>`:''}${mId?`<button data-use-model="${mId}">이 모델로 추론하기</button> <button data-retry-model="${mId}">대기 ${state.images.filter(i=>i.status==='retry').length}장 재추론</button>`:''}</div>`;}).join(''):'<p class="muted">아직 실행한 작업이 없습니다. 모델로 라벨 초안을 만들어보세요.</p>';$$('[data-use-model]').forEach(b=>b.onclick=()=>{$('#model').value=b.dataset.useModel;toast('추론 모델을 선택했습니다.');});$$('[data-retry-model]').forEach(b=>b.onclick=safe(()=>submitInfer(state.images.filter(i=>i.status==='retry').map(i=>i.id),b.dataset.retryModel)));}
async function refresh(){const oldJobs=state.jobs;state=await api('/state');$('#connection').textContent='● 로컬 연결됨';renderList();modelOptions();renderJobs();renderAllInfer();if(current){const row=state.images.find(i=>i.id===current.id);if(row){current={...row};updateHeading();}const completed=state.jobs.some(j=>j.kind==='infer'&&j.state==='complete'&&oldJobs.some(o=>o.id===j.id&&o.state!=='complete')&&j.payload.images.includes(current.id));if(completed&&!dirty&&!dragging&&!polygon.length)await loadImage(current.id,false);}}
async function loadPicture(url){const im=new Image();im.src=url;await im.decode();return im;}
async function readMask(revision,w,h){if(!revision)return new Uint8Array(w*h);const im=await loadPicture('api/revisions/'+revision+'/mask');const c=document.createElement('canvas');c.width=w;c.height=h;const x=c.getContext('2d',{willReadFrequently:true});x.drawImage(im,0,0);const raw=x.getImageData(0,0,w,h).data;return Uint8Array.from({length:w*h},(_,i)=>raw[4*i]);}
async function loadImage(id,fit=true){
 const token=++loadToken,row=state.images.find(i=>i.id===id);if(!row)return;
 const [im,revs]=await Promise.all([loadPicture('api/images/'+id+'/file'),api('/images/'+id+'/revisions')]);
 const saved=revs.find(r=>r.id===row.current),pred=revs.find(r=>r.id===row.prediction);
 const display=pred&&!(saved?.kind==='edited'&&saved.created>=pred.created)?pred.id:row.current;
 const arr=await readMask(display,row.width,row.height);if(token!==loadToken)return;
 current={...row};image=im;mask=arr;activeRevision=row.current;shownRevision=editorRevision=display;
 view='current';dirty=false;undo=[];redo=[];polygon=[];overlay.width=row.width;overlay.height=row.height;
 $('#empty').style.display='none';
 $('#revision').innerHTML='<option value="">현재 편집 화면</option>'+revs.map(r=>`<option value="${r.id}">${r.kind==='prediction'?'예측':r.kind==='edited'?'수정':'원본 라벨'} · ${new Date(r.created*1000).toLocaleString('ko-KR')}${r.model?' · '+(state.models.find(m=>m.id===r.model)?.name??''):''}</option>`).join('');
 updateHeading();setViewButtons();renderList();if(fit)fitImage();else render();
}
function updateHeading(){$('#imageName').textContent=current.name;$('#imageState').textContent=dirty?'수정 중':shownRevision!==activeRevision?'새 예측 · 미검수':names[current.status];const i=filtered().findIndex(x=>x.id===current.id);$('#position').textContent=`${i+1} / ${filtered().length}`;$('#dirtyLabel').textContent=dirty?'● 저장하지 않은 수정':'저장됨';}
async function navigate(id){if(busy)return;busy=true;try{await save();await loadImage(id);}finally{busy=false;}}
async function next(delta=1){const rows=filtered(),i=rows.findIndex(x=>x.id===current?.id);if(rows.length)await navigate(rows[Math.max(0,Math.min(rows.length-1,i+delta))].id);}
function fitImage(){if(!image)return;scale=Math.min(canvas.width/image.width,canvas.height/image.height)*.93;tx=(canvas.width-image.width*scale)/2;ty=(canvas.height-image.height*scale)/2;render();}
function render(){ctx.clearRect(0,0,canvas.width,canvas.height);if(!image)return;ctx.save();ctx.translate(tx,ty);ctx.scale(scale,scale);ctx.drawImage(image,0,0);if(view!=='original'&&$('#overlayBtn').dataset.off!=='true'){const data=oc.createImageData(current.width,current.height);for(let i=0;i<mask.length;i++){const c=mask[i];if(!c)continue;const o=i*4;data.data[o]=c===1?255:c===255?220:0;data.data[o+1]=c===1?128:220;data.data[o+2]=c===1?0:c===255?0:255;data.data[o+3]=255;}oc.putImageData(data,0,0);ctx.globalAlpha=Number($('#opacity').value);ctx.drawImage(overlay,0,0);ctx.globalAlpha=1;}if(polygon.length){ctx.lineWidth=2/scale;ctx.strokeStyle=cls===1?'#ffab56':'#3ef0ff';ctx.beginPath();polygon.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke();for(const [x,y]of polygon){ctx.beginPath();ctx.arc(x,y,3/scale,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();}}ctx.restore();$('#zoomLabel').textContent=Math.round(scale*100)+'%';}
function resize(){const rect=$('#viewport').getBoundingClientRect();canvas.width=rect.width;canvas.height=rect.height;if(image)fitImage();}
new ResizeObserver(resize).observe($('#viewport'));
function canEdit(){
 if(current?.status==='deleted'){toast('삭제 보관함의 데이터는 편집하지 않습니다.');return false;}
 if(!current||view==='original'||$('#revision').value){toast('모델 예측 화면에서 바로 편집하세요. 과거 이력은 아래 버튼으로 가져올 수 있습니다.');return false;}
 return true;
}
function pushUndo(){undo.push(mask.slice());if(undo.length>30)undo.shift();redo=[];}
function changed(){dirty=true;current.status='editing';updateHeading();render();}
function point(e){const r=canvas.getBoundingClientRect();return[(e.clientX-r.left-tx)/scale,(e.clientY-r.top-ty)/scale];}
function dab(x,y){const radius=Number($('#brush').value)/2,value=tool==='erase'?0:cls;for(let yy=Math.max(0,Math.floor(y-radius));yy<Math.min(current.height,y+radius+1);yy++)for(let xx=Math.max(0,Math.floor(x-radius));xx<Math.min(current.width,x+radius+1);xx++)if((xx-x)**2+(yy-y)**2<=radius**2)mask[yy*current.width+xx]=value;}
canvas.tabIndex=0;
canvas.onpointerdown=e=>{if(!image||busy)return;e.preventDefault();canvas.focus({preventScroll:true});canvas.setPointerCapture(e.pointerId);pan=space||e.button===1;last=pan?[e.clientX,e.clientY]:point(e);if(pan){dragging=true;return;}if(!canEdit())return;if(tool==='polygon'){polygon.push(last);render();return;}pushUndo();dragging=true;dab(...last);changed();};
canvas.onpointermove=e=>{if(!dragging)return;if(pan){tx+=e.clientX-last[0];ty+=e.clientY-last[1];last=[e.clientX,e.clientY];render();return;}const p=point(e),distance=Math.hypot(p[0]-last[0],p[1]-last[1]),n=Math.max(1,Math.ceil(distance/Math.max(1,Number($('#brush').value)/4)));for(let i=1;i<=n;i++)dab(last[0]+(p[0]-last[0])*i/n,last[1]+(p[1]-last[1])*i/n);last=p;changed();};
canvas.onpointerup=()=>{dragging=false;pan=false;};canvas.onpointercancel=canvas.onpointerup;
canvas.addEventListener('wheel',e=>{e.preventDefault();if(e.shiftKey){$('#brush').value=Math.max(1,Math.min(80,Number($('#brush').value)+(e.deltaY<0?2:-2)));$('#brush').oninput();return;}if(!image)return;const p=point(e),factor=e.deltaY<0?1.12:1/1.12,nextScale=Math.max(.05,Math.min(15,scale*factor));tx+=(scale-nextScale)*p[0];ty+=(scale-nextScale)*p[1];scale=nextScale;render();},{passive:false});
function finishPolygon(){if(polygon.length<3||!canEdit())return;pushUndo();const c=document.createElement('canvas');c.width=current.width;c.height=current.height;const x=c.getContext('2d');x.beginPath();polygon.forEach(([a,b],i)=>i?x.lineTo(a,b):x.moveTo(a,b));x.closePath();x.fillStyle='white';x.fill();const a=x.getImageData(0,0,c.width,c.height).data;for(let i=0;i<mask.length;i++)if(a[i*4+3]>=128)mask[i]=cls;polygon=[];changed();}
function encodeMask(){const c=document.createElement('canvas');c.width=current.width;c.height=current.height;const x=c.getContext('2d'),d=x.createImageData(c.width,c.height);for(let i=0;i<mask.length;i++){d.data[i*4]=d.data[i*4+1]=d.data[i*4+2]=mask[i];d.data[i*4+3]=255;}x.putImageData(d,0,0);return c.toDataURL('image/png');}
async function save(force=false){if(polygon.length)throw Error('F로 폴리곤을 완성하거나 Esc로 취소하세요.');if(!current||(!dirty&&!force))return;if(view!=='current'||$('#revision').value)throw Error('모델 예측 화면으로 돌아와 저장하세요.');if(polygon.length)throw Error('F로 폴리곤을 완성하거나 Esc로 취소하세요.');const r=await api('/images/'+current.id+'/mask',{mask:encodeMask(),expected_revision:activeRevision});activeRevision=editorRevision=shownRevision=r.revision;current.current=r.revision;dirty=false;updateHeading();}
async function action(status){if(!current||busy)return;if(status==='approved'&&(view!=='current'||$('#revision').value))throw Error('모델 예측 화면에서 승인하세요.');const rows=filtered(),index=rows.findIndex(i=>i.id===current.id),following=rows[index+1]?.id;busy=true;try{await save(status==='approved'&&(!activeRevision||shownRevision!==activeRevision));await api('/images/'+current.id+'/status',{status});await refresh();if(following)await loadImage(following);else{const remaining=filtered().find(i=>i.id!==current.id);if(remaining)await loadImage(remaining.id);else clearEditor();}}finally{busy=false;}}
async function submitInfer(ids,model=$('#model').value){if(!ids.length)throw Error('추론할 이미지를 선택하세요.');if(!model)throw Error('모델을 등록하세요.');await save();await api('/infer',{model,images:ids});toast(`${ids.length}장을 추론 대기열에 넣었습니다.`);await refresh();}
async function setView(mode){
 if(!current)return;await save();view=mode;
 if(mode==='current'){$('#revision').value='';shownRevision=editorRevision;mask=await readMask(editorRevision,current.width,current.height);}
 setViewButtons();render();
}
function setViewButtons(){for(const [id,v]of [['#viewCurrent','current'],['#viewOriginal','original']])$(id).classList.toggle('active',view===v);}

function setTool(t){tool=t;polygon=[];$$('[data-tool]').forEach(b=>b.classList.toggle('active',b.dataset.tool===t));render();}
function setClass(n){cls=n;$$('[data-class]').forEach(b=>b.classList.toggle('active',Number(b.dataset.class)===n));}
function clearEditor(){current=image=mask=null;activeRevision=shownRevision=editorRevision=null;dirty=false;undo=[];redo=[];polygon=[];loadToken++;ctx.clearRect(0,0,canvas.width,canvas.height);$('#imageName').textContent='이 목록의 작업을 마쳤습니다';$('#imageState').textContent='';$('#position').textContent='';$('#dirtyLabel').textContent='';$('#empty').textContent='완료한 이미지는 데이터셋에 보관됩니다. 필터에서 다시 볼 수 있습니다.';$('#empty').style.display='flex';}
$('#filter').onchange=safe(async()=>{
 // Serialize filter navigation behind save/delete/next so an earlier image
 // request cannot replace the newly selected archive or completed view.
 while(busy)await new Promise(resolve=>setTimeout(resolve,20));
 busy=true;
 try{await save();renderList();const rows=filtered();if(!rows.length)clearEditor();else if(!rows.some(i=>i.id===current?.id))await loadImage(rows[0].id);}
 finally{busy=false;}
});$('#search').oninput=renderList;$('#selectAll').onchange=e=>{filtered().forEach(i=>e.target.checked?selected.add(i.id):selected.delete(i.id));renderList();};
$$('[data-class]').forEach(b=>b.onclick=()=>setClass(Number(b.dataset.class)));$$('[data-tool]').forEach(b=>b.onclick=()=>setTool(b.dataset.tool));
$('#brush').oninput=()=>$('#brushValue').textContent=$('#brush').value+' px';$('#opacity').oninput=render;$('#fitBtn').onclick=fitImage;$('#finishBtn').onclick=finishPolygon;
$('#undoBtn').onclick=()=>{if(!canEdit()||!undo.length)return;redo.push(mask.slice());mask=undo.pop();polygon=[];changed();};$('#redoBtn').onclick=()=>{if(!canEdit()||!redo.length)return;undo.push(mask.slice());mask=redo.pop();changed();};
$('#overlayBtn').onclick=()=>{$('#overlayBtn').dataset.off=$('#overlayBtn').dataset.off==='true'?'false':'true';render();};
$('#prevBtn').onclick=safe(()=>next(-1));$('#nextBtn').onclick=safe(()=>next(1));$('#saveBtn').onclick=safe(async()=>{await save(shownRevision!==activeRevision);toast('approve_dataset에 이미지와 라벨을 저장했습니다.');await refresh();});$('#approveBtn').onclick=safe(()=>action('approved'));$('#retryBtn').onclick=safe(()=>action('retry'));$('#excludeBtn').onclick=safe(()=>action('excluded'));
function renderAllInfer(){
 const count=state.images.filter(i=>i.status==='unreviewed').length;
 const running=state.jobs.some(j=>j.kind==='infer'&&['queued','running'].includes(j.state));
 $('#allInfer').disabled=allInferSubmitting||running||!count;
 $('#allInfer').textContent=allInferSubmitting?'전체 재추론 예약 중…':running?'추론 진행 중…':`전체 재추론 · 미검수 ${count.toLocaleString()}장`;
}
$('#allInfer').onclick=safe(async()=>{
 if(allInferSubmitting)return;allInferSubmitting=true;renderAllInfer();
 try{
  await save();await refresh();
  if(state.jobs.some(j=>j.kind==='infer'&&['queued','running'].includes(j.state)))throw Error('현재 추론이 끝난 뒤 다시 실행하세요.');
  const ids=state.images.filter(i=>i.status==='unreviewed').map(i=>i.id);
  if(!ids.length)throw Error('다시 추론할 미검수 이미지가 없습니다.');
  await submitInfer(ids);
 }finally{allInferSubmitting=false;renderAllInfer();}
});
$('#deleteBtn').onclick=safe(async()=>{
 if(!current||busy||current.status==='deleted')return;
 const id=current.id,rows=filtered(),index=rows.findIndex(i=>i.id===id),following=rows[index+1]?.id;
 busy=true;
 try{
  if(polygon.length)throw Error('F로 폴리곤을 완성하거나 Esc로 취소한 뒤 삭제하세요.');
  await save();
  const result=await api('/images/'+id+'/delete',{});
  await refresh();
  if(following&&filtered().some(i=>i.id===following))await loadImage(following);
  else{const next=filtered().find(i=>i.id!==id);if(next)await loadImage(next.id);else clearEditor();}
  toast(`delete_dataset으로 보관했습니다.\n${result.path}`);
 }finally{busy=false;}
});
$('#inferBtn').onclick=safe(()=>submitInfer(current?[current.id]:[]));$('#batchInfer').onclick=safe(()=>submitInfer(selected.size?[...selected]:filtered().map(i=>i.id)));
$('#viewCurrent').onclick=safe(()=>setView('current'));$('#viewOriginal').onclick=safe(()=>setView('original'));
$('#revision').onchange=safe(async()=>{if(dirty){$('#revision').value='';throw Error('먼저 V로 수정본을 저장하세요.');}shownRevision=$('#revision').value||editorRevision;view='current';mask=await readMask(shownRevision,current.width,current.height);setViewButtons();render();});
$('#applyBtn').onclick=safe(async()=>{if(!current||!shownRevision)return;await save();await api('/images/'+current.id+'/apply',{revision:shownRevision});await refresh();await loadImage(current.id,false);toast('선택한 이력을 편집 화면으로 가져왔습니다. 검수 후 승인하세요.');});
$('#importBtn').onclick=()=>{if(!$('#folder').value)$('#folder').value=state.default_folder;$('#importDialog').showModal();};$('#trainBtn').onclick=()=>{const clean=state.models.filter(m=>m.train_eligible);$('#trainModel').value=clean.some(m=>m.id===$('#model').value)?$('#model').value:(clean.at(-1)?.id??'fresh');trainingDefaults();$('#trainDialog').showModal();};$('#registerBtn').onclick=()=>$('#modelDialog').showModal();$$('[data-close]').forEach(b=>b.onclick=()=>b.closest('dialog').close());
$('#importForm').onsubmit=safe(async e=>{e.preventDefault();const button=e.submitter;button.disabled=true;button.textContent='불러오는 중…';try{const r=await api('/import',{path:$('#folder').value,session:$('#session').value});if(r.scanned===0){toast('이 폴더에 지원하는 이미지가 없습니다. 이미지가 있는 폴더를 선택하세요.');return;}$('#importDialog').close();toast(`미라벨 데이터셋에 추가 ${r.added}장 · 중복 ${r.duplicates}장 · 오류 ${r.errors.length}장${r.errors.length?'\n'+r.errors.slice(0,3).join('\n'):''}`);await refresh();}finally{button.disabled=false;button.textContent='데이터셋 추가';}});
$('#trainForm').onsubmit=safe(async e=>{e.preventDefault();await save();const b=e.submitter;b.disabled=true;try{await api('/train',{model:$('#trainModel').value,epochs:Number($('#epochs').value),batch_size:Number($('#batchSize').value),lr:Number($('#lr').value),seed:Number($('#seed').value)});$('#trainDialog').close();toast('승인 데이터를 병합했습니다. eval을 고정한 새 학습을 예약했습니다.');await refresh();}finally{b.disabled=false;}});
$('#modelForm').onsubmit=safe(async e=>{e.preventDefault();await api('/models',{path:$('#modelPath').value,name:$('#modelName').value});$('#modelDialog').close();await refresh();toast('모델을 등록했습니다.');});
$('#exportBtn').onclick=safe(async()=>{await save();const r=await api('/datasets',{});toast(`데이터셋 저장 완료 · train ${r.counts.train} / test ${r.counts.val} / eval ${r.counts.test}\n${r.path}`);});
// Holding Space generates repeated keydown events. Cancel every default scroll,
// including repeats, before the regular one-shot shortcuts ignore those events.
window.addEventListener('keydown',e=>{
 if(e.code!=='Space'||e.ctrlKey||e.metaKey||e.altKey||e.target.closest('input,textarea,select,[contenteditable=true]')||document.querySelector('dialog[open]'))return;
 e.preventDefault();space=true;canvas.style.cursor=dragging&&pan?'grabbing':'grab';
},{capture:true});
window.addEventListener('keydown',e=>{if(e.repeat||e.ctrlKey||e.metaKey||e.altKey||e.target.closest('input,textarea,select,[contenteditable=true]')||document.querySelector('dialog[open]'))return;const map={Digit1:()=>setClass(1),Digit2:()=>setClass(2),KeyQ:()=>next(-1),KeyE:()=>next(1),KeyW:()=>action('approved'),KeyR:()=>action('retry'),KeyA:()=>setTool('brush'),KeyS:()=>setTool('erase'),KeyD:()=>setTool('polygon'),KeyF:finishPolygon,KeyZ:()=>$('#undoBtn').click(),KeyX:()=>$('#redoBtn').click(),KeyC:()=>$('#overlayBtn').click(),KeyV:()=>$('#saveBtn').click(),Escape:()=>{polygon=[];render();}};if(map[e.code]){e.preventDefault();safe(map[e.code])();}});
window.addEventListener('keyup',e=>{if(e.code==='Space'){if(space)e.preventDefault();space=false;canvas.style.cursor='crosshair';}});window.addEventListener('blur',()=>{space=false;dragging=false;});window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='';}});
(async()=>{try{await refresh();if(filtered().length)await loadImage(filtered()[0].id);else clearEditor();setInterval(()=>refresh().catch(()=>$('#connection').textContent='연결 끊김'),3000);}catch(e){toast(e.message);}})();

$('a[href="/"]').onclick=safe(async e=>{e.preventDefault();await save();location.href="/";});
