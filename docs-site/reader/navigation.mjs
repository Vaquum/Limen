import {spring} from './motion.mjs';
import {position,restore,announce} from './place.mjs';
import {initCode} from './code.mjs';
export function initNavigation(){
  const dialog=document.querySelector('[data-reference-dialog]');
  const body=dialog.querySelector('[data-reference-body]');
  const close=dialog.querySelector('[data-reference-close]');
  let opener=null,origin=null,referenceDocument=null,token=0;
  const documentURL=value=>{const url=new URL(value,location.href);return url.origin+url.pathname+url.search;};
  function revealAnchor(){const anchor=decodeURIComponent(location.hash.slice(1));if(anchor){const target=body.querySelector(`#reference-${CSS.escape(anchor)}`);target?.scrollIntoView({block:'start'});target?.focus({preventScroll:true});}else dialog.scrollTop=0;}
  const camera=spring(x=>{dialog.style.transform=`translateX(${x}px)`;dialog.querySelector('.return-rule').style.opacity=String(1-x/Math.max(innerWidth,1)*.5);},innerWidth);
  async function show(link,{updateHistory=true,originOverride=null,referenceURL=link.href}={}){
    const attempt=++token;opener=link;
    origin=originOverride||{url:location.href,place:position(),focus:link.id,scroll:scrollY,width:innerWidth};
    if(updateHistory)history.replaceState({...history.state,readerPlace:origin.place},'',location.href);
    try{
      const response=await fetch(referenceURL);if(!response.ok)throw Error(`Reference returned ${response.status}`);
      const parsed=new DOMParser().parseFromString(await response.text(),'text/html');const main=parsed.querySelector('main article');
      if(!main)throw Error('Reference has no readable main content');if(attempt!==token)return;
      const fragment=document.createElement('div');fragment.append(...[...main.childNodes].map(node=>document.importNode(node,true)));
      const names=new Map([...fragment.querySelectorAll('[id]')].map(node=>[node.id,'reference-'+node.id]));
      for(const node of fragment.querySelectorAll('[id]'))node.id=names.get(node.id);
      for(const node of fragment.querySelectorAll('[href^="#"]')){const id=node.getAttribute('href').slice(1);if(names.has(id)){node.setAttribute('href',new URL('#'+id,referenceURL).href);node.dataset.referenceFragment=id;}}
      const title=document.createElement('h1');title.textContent=parsed.querySelector('h1')?.textContent||'Reference';
      const intro=parsed.querySelector('.hero-lead')?.cloneNode(true);if(intro)intro.className='reference-intro';body.replaceChildren(title,...(intro?[intro]:[]),...fragment.childNodes);
      body.querySelectorAll('[data-reference]').forEach(a=>a.removeAttribute('data-reference'));
      initCode(body);dialog.scrollTop=0;dialog.showModal();referenceDocument=documentURL(referenceURL);
      if(updateHistory)history.pushState({readerReference:true,readerOrigin:origin,readerDepth:1},'',referenceURL);camera.to(0);close.focus({preventScroll:true});revealAnchor();
    }catch(error){announce('The reference could not be opened. Follow its link to retry.');console.error(error);}
  }
  function hide(){++token;if(!dialog.open)return;
    camera.to(innerWidth,()=>{dialog.close();if(origin){if(origin.width===innerWidth)scrollTo({top:origin.scroll,behavior:'instant'});else restore(origin.place);history.replaceState({...history.state,readerPlace:null},'',location.href);opener?.focus({preventScroll:true});}});
  }
  document.addEventListener('click',event=>{
    const fragment=event.target.closest('a[data-reference-fragment]');
    if(dialog.open&&fragment&&body.contains(fragment)&&event.button===0&&!event.metaKey&&!event.ctrlKey&&!event.shiftKey&&!event.altKey){
      event.preventDefault();const url=new URL(location.href);url.hash=fragment.dataset.referenceFragment;
      history.pushState({...history.state,readerReference:true,readerOrigin:origin,readerDepth:history.state.readerDepth+1},'',url);revealAnchor();return;
    }
    const link=event.target.closest('a[data-reference]');
    if(!link||event.button!==0||event.metaKey||event.ctrlKey||event.shiftKey||event.altKey)return;
    event.preventDefault();show(link);
  });
  function returnToChapter(){if(history.state?.readerReference)history.go(-history.state.readerDepth);else hide();}
  close.addEventListener('click',returnToChapter);
  dialog.addEventListener('cancel',event=>{event.preventDefault();returnToChapter();});
  addEventListener('popstate',event=>{
    if(event.state?.readerReference){
      if(dialog.open&&referenceDocument===documentURL(location.href)){revealAnchor();return;}
      const link=document.getElementById(event.state.readerOrigin.focus);
      if(link)show(link,{updateHistory:false,originOverride:event.state.readerOrigin,referenceURL:location.href});
    }
    else hide();
  });
  for(const name of ['wheel','touchstart','pointerdown'])dialog.addEventListener(name,()=>camera.finish(),{passive:true});
}
