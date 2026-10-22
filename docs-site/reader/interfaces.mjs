import {announce} from './place.mjs';

const choices=['both','python','compiled'];
export function initInterfaces(){
  const article=document.querySelector('main article'),control=document.querySelector('[data-interface-controls]');
  if(!article||!control)return;
  // Only explicit authoring sections and executed examples are interface-specific.
  for(const heading of [...article.querySelectorAll('h2,h3,h4')].reverse()){
    const text=heading.textContent,kind=/\bCompiled\b/.test(text)?'compiled':/\bPython\b/.test(text)?'python':null;
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
  function choose(value,{save=true,move=false}={}){
    if(!choices.includes(value))return;
    const anchor=[...article.querySelectorAll('[data-passage]')].filter(node=>node.getClientRects().length&&node.getBoundingClientRect().top<=130).at(-1);
    const top=anchor?.getBoundingClientRect().top||0;
    document.body.dataset.interface=value;
    for(const node of article.querySelectorAll('[data-research-interface]'))node.hidden=value!=='both'&&node.dataset.researchInterface!==value;
    for(const button of control.querySelectorAll('button'))button.setAttribute('aria-pressed',String(button.dataset.interfaceChoice===value));
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
    if(!location.hash&&saved){const place=JSON.parse(saved);if(choices.includes(place.interface))preferred=place.interface;}
  }catch(error){announce('The saved interface choice could not be read.');console.error(error);}
  if(!location.hash&&choices.includes(history.state?.readerPlace?.interface))preferred=history.state.readerPlace.interface;
  const target=location.hash?document.getElementById(decodeURIComponent(location.hash.slice(1))):null;
  const linked=target?.closest('[data-research-interface]');
  choose(linked?.dataset.researchInterface||preferred,{save:false});
  control.hidden=false;
  control.addEventListener('click',event=>{const button=event.target.closest('[data-interface-choice]');if(button)choose(button.dataset.interfaceChoice,{move:true});});
}
