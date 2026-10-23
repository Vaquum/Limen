import {draftStore} from './drafts.mjs';
const $=selector=>document.querySelector(selector),frame=$('[data-frame]'),status=$('[data-status]'),params=new URLSearchParams(location.search),pr=params.get('pr'),suffix=pr?'?pr='+pr:'';
let auth,store,busy=false,page;
const say=text=>status.textContent=text;
async function api(path,body){const reply=await fetch('/limen/edit/api/'+path+suffix,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{}),data=await reply.json();if(!reply.ok)throw Error(data.error);return data;}
const send=(type,extra={})=>frame.contentWindow?.postMessage({type,...extra},location.origin);
function update(){if(!store)return;const items=store.heads();$('[data-count]').textContent=items.length;$('[data-commit]').disabled=busy||(!items.length&&auth.workspace.operation?.kind!=='commit')||store.conflicts();$('[data-pr]').disabled=busy||!auth.workspace.created||items.length>0;$('[data-review]').disabled=false;$('[data-edit]').disabled=busy;if(store.conflicts())say('Competing tabs edited the same passage. Review and Undo the unwanted version before committing.');}
function review(){const list=$('[data-drafts]');list.replaceChildren();for(const item of store.heads()){const li=document.createElement('li'),name=document.createElement('p'),before=document.createElement('pre'),after=document.createElement('pre'),undo=document.createElement('button');name.textContent=item.patch.path;before.textContent=item.patch.originalText;after.textContent=item.patch.newMarkdown;undo.textContent='Undo this draft';undo.onclick=()=>{store.undo(item.id);review();update();frame.contentWindow.location.reload();};li.append(name,before,after,undo);list.append(li);}if(!$('[data-review-dialog]').open)$('[data-review-dialog]').showModal();}
async function action(run){busy=true;update();try{await run();}catch(error){say(error.message+' Drafts are retained.');}finally{busy=false;update();}}
async function connect(){
  try{
    auth=await api('session');store=draftStore(auth.user.id,auth.workspace.id);$('[data-user]').textContent=auth.user.login+' · '+auth.workspace.branch;$('[data-signin]').hidden=true;$('[data-logout]').hidden=false;frame.hidden=false;frame.src='/limen/edit/view'+(params.get('route')||'/')+suffix;$('[data-edit]').disabled=false;
    say('Authoring '+auth.workspace.head.slice(0,8)+'. Branch commits receive PR checks after Open PR.');update();
    if(auth.workspace.operation)say('A remote operation needs reconciliation. Use '+(auth.workspace.operation.kind==='commit'?'Commit':'Open PR')+' to resume it.');
  }catch(error){say(error.message);const target=new URL('/limen/edit/auth/start',location.origin);target.searchParams.set('route',params.get('route')||'/');if(pr)target.searchParams.set('pr',pr);$('[data-signin]').href=target.href;}
}
addEventListener('message',event=>{
  if(event.origin!==location.origin||event.source!==frame.contentWindow||!store)return;
  try{
    if(event.data.type==='page'){page=event.data.page;if(page.workspace!==auth.workspace.id)throw Error('The frame belongs to another delivery');const url=new URL(location.href);url.searchParams.set('route',page.route);history.replaceState(null,'',url);send('drafts',{items:store.heads()});send('edit',{enabled:$('[data-edit]').checked});}
    if(event.data.type==='draft'){store.save(event.data.patch,event.data.html);say('Draft saved in this browser.');update();}
    if(event.data.type==='error'||event.data.type==='conflict')say(event.data.message);
    if(event.data.type==='active')$('[data-undo]').disabled=!event.data.active;
  }catch(error){say('Draft storage failed: '+error.message);}
});
$('[data-edit]').onchange=()=>send('edit',{enabled:$('[data-edit]').checked});$('[data-undo]').onpointerdown=event=>{event.preventDefault();send('undo-active');};$('[data-undo]').onclick=()=>send('undo-active');$('[data-review]').onclick=()=>{send('finish');setTimeout(review,0);};$('[data-close]').onclick=()=>$('[data-review-dialog]').close();
$('[data-commit]').onclick=()=>{send('finish');setTimeout(()=>action(async()=>{
  if(store.conflicts())throw Error('Resolve competing drafts first');const items=store.heads(),existing=auth.workspace.operation;
  const operation=existing?.kind==='commit'?existing.id:crypto.randomUUID();
  const payload={workspace:auth.workspace.id,expectedHead:auth.workspace.head,operation,patches:items.map(item=>item.patch)};
  auth.workspace=await api('commit',payload);store.confirmed(auth.workspace.receipt.drafts);say('Committed '+auth.workspace.receipt.head.slice(0,8)+' to origin/'+auth.workspace.branch+'.');frame.contentWindow.location.reload();
}),0);};
$('[data-pr]').onclick=()=>action(async()=>{const result=await api('pr',{workspace:auth.workspace.id,operation:auth.workspace.operation?.kind==='pr'?auth.workspace.operation.id:crypto.randomUUID(),issue:Number($('[data-issue]').value)||undefined});auth.workspace.pr=result.number;say('PR #'+result.number+' contains the branch commits. Finish checks and review in GitHub.');const link=document.createElement('a');link.href=result.url;link.textContent='Open PR #'+result.number;link.target='_blank';link.rel='noopener';status.append(' ',link);});
$('[data-download]').onclick=()=>{const blob=new Blob([JSON.stringify(store.heads(),null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download='limen-prose-drafts.json';link.click();URL.revokeObjectURL(url);};
$('[data-logout]').onclick=()=>action(async()=>{
  send('finish');await new Promise(resolve=>setTimeout(resolve,0));const items=store.heads();if(items.length&&!confirm('Discard '+items.length+' local drafts and sign out? Cancel to commit or download them first.'))return;
  const result=await api('logout',{});for(const item of items)store.undo(item.id);frame.hidden=true;auth=null;say(result.revoked?'Signed out; GitHub authorization revoked.':'Signed out. GitHub token revocation failed; revoke the App authorization in GitHub settings.');$('[data-signin]').hidden=false;$('[data-logout]').hidden=true;for(const element of document.querySelectorAll('button,input'))element.disabled=true;
});
addEventListener('storage',()=>{try{update();if(store)send('drafts',{items:store.heads()});}catch(error){say('Draft storage failed: '+error.message);}});
addEventListener('beforeunload',event=>{if(busy){event.preventDefault();event.returnValue='';}});
connect();
