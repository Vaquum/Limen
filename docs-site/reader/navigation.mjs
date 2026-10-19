import {spring} from './motion.mjs';
import {position,restore,announce} from './place.mjs';
import {initCode} from './code.mjs';
export function initNavigation(){
  const dialog=document.querySelector('[data-reference-dialog]');
  const body=dialog.querySelector('[data-reference-body]');
  const close=dialog.querySelector('[data-reference-close]');
  let opener=null,origin=null,token=0;
  const camera=spring(x=>{dialog.style.transform=`translateX(${x}px)`;dialog.querySelector('.return-rule').style.opacity=String(1-x/Math.max(innerWidth,1)*.5);},innerWidth);
  async function show(link,{updateHistory=true,originOverride=null}={}){
    const attempt=++token;opener=link;
    origin=originOverride||{url:location.href,place:position(),focus:link.id};
    if(updateHistory)history.replaceState({...history.state,readerPlace:origin.place},'',location.href);
    try{
      const response=await fetch(link.href);if(!response.ok)throw Error(`Reference returned ${response.status}`);
      const parsed=new DOMParser().parseFromString(await response.text(),'text/html');const main=parsed.querySelector('main');
      if(!main)throw Error('Reference has no readable main content');if(attempt!==token)return;
      const fragment=document.createElement('div');fragment.append(...[...main.childNodes].map(node=>document.importNode(node,true)));
      const names=new Map([...fragment.querySelectorAll('[id]')].map(node=>[node.id,'reference-'+node.id]));
      for(const node of fragment.querySelectorAll('[id]'))node.id=names.get(node.id);
      for(const node of fragment.querySelectorAll('[href^="#"]')){const id=node.getAttribute('href').slice(1);if(names.has(id))node.setAttribute('href','#'+names.get(id));}
      const title=document.createElement('h1');title.textContent=parsed.querySelector('h1')?.textContent||'Reference';
      body.replaceChildren(title,...fragment.childNodes);
      body.querySelectorAll('[data-reference]').forEach(a=>a.removeAttribute('data-reference'));
      initCode(body);dialog.scrollTop=0;dialog.showModal();
      if(updateHistory)history.pushState({readerReference:true,readerOrigin:origin},'',link.href);camera.to(0);close.focus({preventScroll:true});
      const anchor=new URL(link.href).hash.slice(1);if(anchor)body.querySelector(`#reference-${CSS.escape(anchor)}`)?.scrollIntoView({block:'start'});
    }catch(error){announce('The reference could not be opened. Follow its link to retry.');console.error(error);}
  }
  function hide(){++token;if(!dialog.open)return;
    camera.to(innerWidth,()=>{dialog.close();if(origin){restore(origin.place);opener?.focus({preventScroll:true});}});
  }
  document.addEventListener('click',event=>{
    const link=event.target.closest('a[data-reference]');
    if(!link||event.button!==0||event.metaKey||event.ctrlKey||event.shiftKey||event.altKey)return;
    event.preventDefault();show(link);
  });
  close.addEventListener('click',()=>{if(history.state?.readerReference)history.back();else hide();});
  dialog.addEventListener('cancel',event=>{event.preventDefault();if(history.state?.readerReference)history.back();else hide();});
  addEventListener('popstate',event=>{
    if(event.state?.readerReference){const link=[...document.querySelectorAll('a[data-reference]')].find(a=>a.href===location.href);if(link)show(link,{updateHistory:false,originOverride:event.state.readerOrigin});}
    else hide();
  });
  for(const name of ['wheel','touchstart','pointerdown'])dialog.addEventListener(name,()=>camera.finish(),{passive:true});
}
