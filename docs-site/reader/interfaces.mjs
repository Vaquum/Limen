import {announce,navigationType,resuming,restore} from './place.mjs';
import {revealCodeRange} from './code.mjs';

const choices=['both','python','compiled'];
export function initInterfaces(){
  const article=document.querySelector('main article'),control=document.querySelector('[data-interface-controls]');
  if(!article||!control)return;
  // Only explicit authoring sections and executed examples are interface-specific.
  for(const heading of [...article.querySelectorAll('h2,h3,h4')].reverse()){
    const text=heading.textContent,kind=heading.hasAttribute('data-heading-interface')?(heading.dataset.headingInterface==='plain'?null:heading.dataset.headingInterface):/\bCompiled\b/.test(text)?'compiled':/\bPython\b/.test(text)?'python':null;
    if(!kind)continue;
    const section=document.createElement('section');section.dataset.researchInterface=kind;
    heading.before(section);section.append(heading);
    while(section.nextElementSibling){const next=section.nextElementSibling;
      if(/^H[1-4]$/.test(next.tagName)&&Number(next.tagName[1])<=Number(heading.tagName[1]))break;
      if(next.matches('section[data-research-interface]')&&Number(next.firstElementChild.tagName[1])<=Number(heading.tagName[1]))break;
      section.append(next);
    }
  }
  const key=`reader:${document.body.dataset.product}:interface`;
  if(!article.querySelector('[data-research-interface]'))return;
  const available=new Set([...article.querySelectorAll('[data-research-interface]')].map(node=>node.dataset.researchInterface));
  function choose(value,{save=true,move=false,record=true}={}){
    if(!choices.includes(value)||(value!=='both'&&!available.has(value)))return;
    const anchor=[...article.querySelectorAll('[data-passage]')].filter(node=>node.getClientRects().length&&node.getBoundingClientRect().top<=130).at(-1);
    const top=anchor?.getBoundingClientRect().top||0;
    document.body.dataset.interface=value;
    for(const node of article.querySelectorAll('[data-research-interface]'))node.hidden=value!=='both'&&node.dataset.researchInterface!==value;
    for(const button of control.querySelectorAll('button')){button.setAttribute('aria-pressed',String(button.dataset.interfaceChoice===value));button.disabled=button.dataset.interfaceChoice!=='both'&&!available.has(button.dataset.interfaceChoice);}
    if(record)history.replaceState({...history.state,readerInterface:value},'',location.href);
    if(move&&anchor){
      if(anchor.getClientRects().length)scrollBy({top:anchor.getBoundingClientRect().top-top,behavior:'instant'});
      else {
        const section=anchor.closest('section[data-research-interface]');
        const previous=section?[...article.querySelectorAll(`section[data-research-interface="${section.dataset.researchInterface}"]`)]:[];
        const counterparts=[...article.querySelectorAll(`section[data-research-interface="${value}"]`)];
        const corresponding=counterparts[previous.indexOf(section)]||counterparts[0];
        corresponding?.scrollIntoView({block:'start'});
      }
    }
    if(save){try{localStorage.setItem(key,value);}catch(error){announce('The interface choice could not be saved.');console.error(error);}dispatchEvent(new Event('reader-interface-change'));}
  }
  let preferred=document.body.dataset.interface;
  try{preferred=localStorage.getItem(key)||preferred;
    const saved=localStorage.getItem(`reader:${document.body.dataset.product}:${document.body.dataset.content}`);
    if(resuming()&&!location.hash&&saved){const place=JSON.parse(saved);if(choices.includes(place.interface))preferred=place.interface;}
  }catch(error){announce('The saved interface choice could not be read.');console.error(error);}
  if(navigationType()==='back_forward'&&!location.hash&&choices.includes(history.state?.readerPlace?.interface))preferred=history.state.readerPlace.interface;
  if(preferred!=='both'&&!available.has(preferred))preferred=[...available][0];
  const target=location.hash?document.getElementById(decodeURIComponent(location.hash.slice(1)).replace(/-L(\d+)-L\d+$/,'-L$1')):null;
  const linked=target?.closest('[data-research-interface]');
  const entry=navigationType()==='back_forward'&&choices.includes(history.state?.readerInterface)?history.state.readerInterface:null;
  choose(entry||linked?.dataset.researchInterface||preferred,{save:false});
  document.addEventListener('click',event=>{
    const link=event.target.closest('a[href]');
    if(!link||event.defaultPrevented||event.button!==0||event.metaKey||event.ctrlKey||event.shiftKey||event.altKey||document.querySelector('dialog[open]'))return;
    const url=new URL(link.href);
    if(url.origin!==location.origin||url.pathname!==location.pathname||url.search!==location.search||!url.hash)return;
    const target=document.getElementById(decodeURIComponent(url.hash.slice(1)).replace(/-L(\d+)-L\d+$/,'-L$1'));
    const linked=target?.closest('[data-research-interface]');
    if(linked)choose(linked.dataset.researchInterface,{save:false,record:url.hash===location.hash});
  });
  addEventListener('hashchange',()=>{
    if(history.state?.readerReference||document.querySelector('dialog[open]'))return;
    if(!choices.includes(history.state?.readerInterface)){
      const target=document.getElementById(decodeURIComponent(location.hash.slice(1)).replace(/-L(\d+)-L\d+$/,'-L$1'));
      const linked=target?.closest('[data-research-interface]');
      if(linked)choose(linked.dataset.researchInterface,{save:false});
      revealCodeRange();
      if(target){target.scrollIntoView({block:target.classList.contains('code-line')?'center':'start'});target.focus({preventScroll:true});}
    }else revealCodeRange({scroll:false});
    history.replaceState({...history.state,readerInterface:document.body.dataset.interface},'',location.href);
  });
  addEventListener('popstate',event=>{
    if(event.state?.readerReference||document.querySelector('dialog[open]'))return;
    if(choices.includes(event.state?.readerInterface)){
      choose(event.state.readerInterface,{save:false,record:false});
      if(event.state.readerPlace)requestAnimationFrame(()=>restore(event.state.readerPlace));
    }
  });
  control.hidden=false;
  control.addEventListener('click',event=>{const button=event.target.closest('[data-interface-choice]');if(button)choose(button.dataset.interfaceChoice,{move:true});});
}
