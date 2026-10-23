import {inlineMarkdown} from './inline.mjs';
const data=JSON.parse(document.querySelector('#editor-source').content.textContent),mapped=new Map(data.passages.map(item=>[item.id,item]));
let enabled=false,active=null;
const send=(type,extra={})=>parent.postMessage({type,...extra},location.origin);
function finish(cancel=false){
  if(!active)return;const {element,html,passage,links}=active;
  try{
    if(cancel)element.innerHTML=html;
    else if(element.innerHTML!==html){const newMarkdown=inlineMarkdown(element,{links});send('draft',{patch:{path:data.path,baseBlobSha:data.blob,sourceRange:passage.sourceRange,originalText:passage.originalText,newMarkdown},html:element.innerHTML});element.dataset.drafted='true';}
  }catch(error){element.innerHTML=html;send('error',{message:error.message});}
  element.contentEditable='false';element.removeAttribute('role');active=null;send('active',{active:false});
}
function begin(element){
  if(!enabled||active?.element===element)return;finish();const passage=mapped.get(element.dataset.editId);if(!passage)return;
  const links=[...element.querySelectorAll('a')].map(link=>({href:link.dataset.sourceHref||link.getAttribute('href'),title:link.getAttribute('title')}));
  active={element,passage,html:element.innerHTML,links};element.contentEditable='true';element.setAttribute('role','textbox');element.setAttribute('aria-label','Edit '+passage.type);element.focus({preventScroll:true});send('active',{active:true});
}
document.addEventListener('click',event=>{const element=event.target.closest('[data-edit-id]');if(enabled&&element){event.preventDefault();begin(element);}});
document.addEventListener('keydown',event=>{
  if(!active){if(enabled&&event.key==='Enter'&&event.target.matches('[data-edit-id]')){event.preventDefault();begin(event.target);}return;}
  if(event.isComposing)return;
  if(event.key==='Escape'){event.preventDefault();finish(true);}
  if(event.key==='Enter'){
    event.preventDefault();if(event.shiftKey&&active.passage.type==='paragraph'){
      const selection=getSelection(),range=selection.getRangeAt(0);range.deleteContents();const br=document.createElement('br');range.insertNode(br);range.setStartAfter(br);range.collapse(true);selection.removeAllRanges();selection.addRange(range);
    }else finish();
  }
});
document.addEventListener('paste',event=>{if(!active)return;event.preventDefault();const range=getSelection().getRangeAt(0),text=document.createTextNode(event.clipboardData.getData('text/plain').replace(/\r?\n/g,' '));range.deleteContents();range.insertNode(text);range.setStartAfter(text);range.collapse(true);});
document.addEventListener('drop',event=>{if(active)event.preventDefault();});
document.addEventListener('focusout',event=>{if(active&&event.target===active.element)finish();});
addEventListener('message',async event=>{
  if(event.origin!==location.origin||event.source!==parent)return;
  const message=event.data;
  if(message.type==='edit'){finish();enabled=message.enabled;document.body.classList.toggle('prose-editing',enabled);for(const element of document.querySelectorAll('[data-edit-id]')){element.tabIndex=enabled?0:-1;element.setAttribute('aria-label',enabled?'Edit '+element.dataset.editType:element.dataset.editType);}}
  if(message.type==='finish')finish();if(message.type==='undo-active')finish(true);
  if(message.type==='drafts'){
    for(const item of message.items){if(item.patch.path!==data.path)continue;const passage=data.passages.find(passage=>passage.baseBlobSha===item.patch.baseBlobSha&&passage.sourceRange.start===item.patch.sourceRange.start&&passage.originalText===item.patch.originalText);if(!passage){send('conflict',{message:'A draft no longer matches this source blob. It remains available in Review.'});continue;}
      const element=document.querySelector(`[data-edit-id="${passage.id}"]`);if(!element)continue;
      try{const reply=await fetch('/limen/edit/api/inline'+(data.pr?'?pr='+data.pr:''),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({patch:item.patch})}),result=await reply.json();if(!reply.ok)throw Error(result.error);element.innerHTML=result.html;element.dataset.drafted='true';}catch(error){send('error',{message:error.message});}
    }
  }
});
// Navigation must retain active edits before the frame unloads.
document.addEventListener('click',event=>{if(!enabled&&event.target.closest('a'))finish();},true);
addEventListener('pagehide',()=>finish());
send('page',{page:data});
