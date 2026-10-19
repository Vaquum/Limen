const product=document.body.dataset.product;
const content=document.body.dataset.content;
const key=`reader:${product}:${content}`;
const status=document.querySelector('[data-status]');
let storageAvailable=true;
export function announce(message){if(status)status.textContent=message;}
export function position(){
  const passages=[...document.querySelectorAll('main article [data-passage]')];
  const line=Math.min(130,innerHeight*.2);
  const active=passages.filter(p=>p.getBoundingClientRect().top<=line).at(-1);
  if(!active)return {passage:'opening',ratio:0,interface:document.body.dataset.interface||'both'};
  const rect=active.getBoundingClientRect();
  return {passage:active.id,ratio:Math.max(0,Math.min(1,(line-rect.top)/Math.max(1,rect.height))),interface:document.body.dataset.interface||'both'};
}
export function restore(place){
  if(!place||typeof place.passage!=='string'||typeof place.ratio!=='number'||!Number.isFinite(place.ratio)||place.ratio<0||place.ratio>1){announce('Saved reading position is invalid.');return false;}
  if(place.passage==='opening'){scrollTo(0,0);return true;}
  const el=document.getElementById(place.passage);
  if(!el?.matches('[data-passage]')){announce('The saved passage is no longer in this revision.');return false;}
  const rect=el.getBoundingClientRect();scrollTo({top:scrollY+rect.top+rect.height*place.ratio-Math.min(130,innerHeight*.2),behavior:'instant'});
  document.body.dataset.interface=place.interface||'both';return true;
}
export function initPlace(){
  const resumed=history.state?.readerPlace;
  if(resumed&&!location.hash)restore(resumed);
  else if(!location.hash){try{const raw=localStorage.getItem(key);if(raw)restore(JSON.parse(raw));}catch(error){storageAvailable=false;announce('Browser storage could not be read. Reading position is unavailable.');console.error(error);}}
  let frame=0, moved=false;
  const progress=document.querySelector('.reading-progress');
  function save(){frame=0;if(document.querySelector('dialog[open]'))return;
    const value=position();const maximum=document.documentElement.scrollHeight-innerHeight;
    if(progress)progress.style.setProperty('--progress',`${maximum>0?Math.max(0,Math.min(100,scrollY/maximum*100)):100}%`);
    if(!moved||!storageAvailable)return;
    try{localStorage.setItem(key,JSON.stringify(value));}catch(error){storageAvailable=false;announce('Browser storage could not be written. Reading position is not being saved.');console.error(error);}
  }
  for(const name of ['wheel','touchmove'])addEventListener(name,()=>{moved=true;},{passive:true});
  addEventListener('keydown',event=>{if(!event.target.closest('input,textarea,select,[contenteditable]')&&['PageDown','PageUp','ArrowDown','ArrowUp','Home','End',' '].includes(event.key))moved=true;});
  addEventListener('scroll',()=>{if(!frame)frame=requestAnimationFrame(save);},{passive:true});
  addEventListener('pagehide',()=>{if(moved)save();});save();
}
