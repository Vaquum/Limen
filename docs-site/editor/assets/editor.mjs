import {draftStore,writerStores,clearWriter} from './drafts.mjs';
const $=selector=>document.querySelector(selector),frame=$('[data-frame]'),status=$('[data-status]'),params=new URLSearchParams(location.search),pr=params.get('pr'),suffix=pr?'?pr='+pr:'';
let auth,store,busy=false,page,ready=false,restoreRequest,previewFailed=false;
function restoreDrafts(){restoreRequest=crypto.randomUUID();send('drafts',{items:store.heads(),request:restoreRequest});}const finishing=new Map();
function flush(){if(!ready)return Promise.resolve();const request=crypto.randomUUID();return new Promise((resolve,reject)=>{const timer=setTimeout(()=>{finishing.delete(request);reject(Error('The active passage could not be saved'));},5000);finishing.set(request,error=>{clearTimeout(timer);if(error)reject(Error(error));else resolve();});send('finish',{request});});}
function acknowledge(){for(const receipt of Object.values(auth.workspace.receipts||{}))store.confirmed(receipt.drafts);}
const say=text=>status.textContent=text;
async function api(path,body){const reply=await fetch('/limen/edit/api/'+path+suffix,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{}),data=await reply.json();if(!reply.ok)throw Error(data.error);return data;}
const send=(type,extra={})=>frame.contentWindow?.postMessage({type,...extra},location.origin);
frame.addEventListener('load',()=>{
  const doc=frame.contentDocument;if(!auth||frame.hidden||doc?.querySelector('#editor-source'))return;
  ready=false;previewFailed=true;let message='The authoring page could not be loaded';
  if(doc?.contentType==='application/json'){try{message=JSON.parse(doc.querySelector('pre')?.textContent||doc.body.textContent).error||message;}catch(error){message+=': '+error.message;}}
  say(message);update();
});
let lastEditState;
function update(){if(!store||!auth)return;const editing=$('[data-edit]').checked&&!busy&&ready;frame.inert=busy||!ready;if(editing!==lastEditState){lastEditState=editing;send('edit',{enabled:editing});}const items=store.heads();$('[data-count]').textContent=items.length;$('[data-commit]').disabled=busy||!ready||(!items.length&&auth.workspace.operation?.kind!=='commit')||store.conflicts();$('[data-pr]').disabled=busy||!ready||!auth.workspace.created||items.length>0;$('[data-review]').disabled=busy;$('[data-refresh]').disabled=busy;$('[data-logout]').disabled=busy;$('[data-edit]').disabled=busy||!ready;if(store.conflicts())say('Competing tabs edited the same passage. Review and Undo the unwanted version before committing.');}
function reloadPreview(){ready=false;update();frame.contentWindow.location.reload();}
function review(){const list=$('[data-drafts]');list.replaceChildren();for(const delivery of writerStores(auth.user.id))for(const item of delivery.store.heads()){const li=document.createElement('li'),name=document.createElement('p'),before=document.createElement('pre'),after=document.createElement('pre'),undo=document.createElement('button');name.textContent=item.patch.path+(delivery.workspace===auth.workspace.id?'':' · previous delivery '+delivery.workspace);before.textContent=item.patch.originalText;after.textContent=item.patch.newMarkdown;undo.textContent='Undo this draft';undo.onclick=()=>{delivery.store.undo(item.id);review();update();reloadPreview();};li.append(name,before,after,undo);list.append(li);}if(!$('[data-review-dialog]').open)$('[data-review-dialog]').showModal();}
async function action(run){if(busy)return;busy=true;update();try{await run();}catch(error){say(error.message+' Drafts are retained.');}finally{busy=false;update();}}
async function connect(){
  try{
    auth=await api('session');localStorage.removeItem(`limen-editor:${auth.user.id}:signed-out`);store=draftStore(auth.user.id,auth.workspace.id);acknowledge();$('[data-user]').textContent=auth.user.login+' · '+auth.workspace.branch;$('[data-signin]').hidden=true;$('[data-logout]').hidden=false;frame.hidden=false;frame.src='/limen/edit/view'+(params.get('route')||'/')+suffix;$('[data-edit]').disabled=false;
    say('Authoring '+auth.workspace.head.slice(0,8)+'. Branch commits receive PR checks after Open PR.');update();
    if(auth.workspace.operation)say('A remote operation needs reconciliation. Use '+(auth.workspace.operation.kind==='commit'?'Commit':'Open PR')+' to resume it.');
  }catch(error){say(error.message);const target=new URL('/limen/edit/auth/start',location.origin);target.searchParams.set('route',params.get('route')||'/');if(pr)target.searchParams.set('pr',pr);$('[data-signin]').href=target.href;}
}
addEventListener('limen-editor-loading',()=>{ready=false;update();});
addEventListener('limen-editor-draft',event=>{const data=event.detail;try{if(!auth||data.workspace!==auth.workspace.id||data.path!==page?.path)throw Error('The passage belongs to another delivery');const item=store.save(data.patch,data.previous);data.saved=item.id;say('Draft saved in this browser.');update();}catch(error){data.error=error.message;say('Draft storage failed: '+error.message);}});
addEventListener('message',event=>{
  if(event.origin!==location.origin||event.source!==frame.contentWindow||!store)return;
  try{
    if(event.data.type==='page'){if(previewFailed){say('Authoring preview restored. Drafts remain in Review.');previewFailed=false;}page=event.data.page;if(page.workspace!==auth.workspace.id)throw Error('The frame belongs to another delivery');const url=new URL(location.href);url.searchParams.set('route',page.route);history.replaceState(null,'',url);ready=false;restoreDrafts();update();}
    if(event.data.type==='drafts-applied'&&event.data.request===restoreRequest){ready=true;frame.dataset.restored=page.head;update();}
    if(event.data.type==='finished'){finishing.get(event.data.request)?.(event.data.error);finishing.delete(event.data.request);}
    if(event.data.type==='edit-rejected'){$('[data-edit]').checked=event.data.enabled;lastEditState=event.data.enabled;}
    if(event.data.type==='draft'){const item=store.save(event.data.patch,event.data.previous);send('draft-saved',{passageId:event.data.passageId,id:item.id});say('Draft saved in this browser.');update();}
    if(event.data.type==='error'||event.data.type==='conflict')say(event.data.message);
    if(event.data.type==='active'){$('[data-undo]').disabled=!event.data.active;if(!event.data.active)restoreDrafts();}
  }catch(error){say('Draft storage failed: '+error.message);}
});
$('button[data-refresh]').onclick=()=>action(async()=>{await flush();const updated=await api('session');if(updated.workspace.id!==auth.workspace.id){say('The previous delivery is closed. Download or discard its remaining drafts in Review, then reload for the next delivery.');return;}auth=updated;acknowledge();if(!auth.workspace.operation){for(const item of store.heads()){try{const result=await api('reanchor',{patch:item.patch});if(result.patch.baseBlobSha!==item.patch.baseBlobSha){store.save(result.patch,[]);store.undo(item.id);}}catch(error){say(error.message+' The original draft remains in Review.');}}}reloadPreview();update();});
$('[data-edit]').onchange=()=>update();$('[data-undo]').onpointerdown=event=>{event.preventDefault();send('undo-active');};$('[data-undo]').onclick=()=>send('undo-active');$('[data-review]').onclick=()=>action(async()=>{await flush();review();});$('[data-close]').onclick=()=>$('[data-review-dialog]').close();
$('[data-commit]').onclick=()=>action(async()=>{
  await flush();
  if(store.conflicts())throw Error('Resolve competing drafts first');const items=store.heads(),existing=auth.workspace.operation;
  const operation=existing?.kind==='commit'?existing.id:crypto.randomUUID();
  const payload={workspace:auth.workspace.id,expectedHead:auth.workspace.head,operation,patches:items.map(item=>item.patch)};
  auth.workspace=await api('commit',payload);store.confirmed(auth.workspace.receipt.drafts);say('Committed '+auth.workspace.receipt.head.slice(0,8)+' to origin/'+auth.workspace.branch+'.');reloadPreview();
});
$('[data-pr]').onclick=()=>action(async()=>{const result=await api('pr',{workspace:auth.workspace.id,operation:auth.workspace.operation?.kind==='pr'?auth.workspace.operation.id:crypto.randomUUID(),issue:Number($('[data-issue]').value)||undefined});auth.workspace.pr=result.number;say('PR #'+result.number+' contains the branch commits. Finish checks and review in GitHub.');const link=document.createElement('a');link.href=result.url;link.textContent='Open PR #'+result.number;link.target='_blank';link.rel='noopener';status.append(' ',link);});
$('[data-download]').onclick=()=>{const blob=new Blob([JSON.stringify(writerStores(auth.user.id).map(item=>({workspace:item.workspace,drafts:item.store.heads()})),null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download='limen-prose-drafts.json';link.click();URL.revokeObjectURL(url);};
$('[data-logout]').onclick=()=>action(async()=>{
  await flush();const items=writerStores(auth.user.id).flatMap(item=>item.store.heads());if(items.length&&!confirm('Discard all '+items.length+' browser drafts for this writer and sign out? Cancel to commit or download them first.'))return;
  const result=await api('logout',{});clearWriter(auth.user.id);frame.hidden=true;auth=null;store=null;say(result.revoked?'Signed out; GitHub authorization revoked.':'Signed out. GitHub token revocation failed (HTTP '+(result.revocationStatus||'unavailable')+').' );$('[data-signin]').hidden=false;$('[data-logout]').hidden=true;for(const element of document.querySelectorAll('button,input'))element.disabled=true;
});
addEventListener('storage',event=>{if(auth&&event.key===`limen-editor:${auth.user.id}:signed-out`){frame.hidden=true;auth=null;store=null;say('Signed out in another tab.');return;}if(!store||!event.key?.startsWith(store.prefix))return;try{update();restoreDrafts();}catch(error){say('Draft storage failed: '+error.message);}});
addEventListener('beforeunload',event=>{const detail={saved:true};frame.contentWindow?.dispatchEvent(new CustomEvent('limen-editor-flush',{detail}));if(busy||!detail.saved){event.preventDefault();event.returnValue='';}});
connect();
