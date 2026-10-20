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
  const next=passages[passages.indexOf(active)+1];
  const span=next?next.getBoundingClientRect().top-rect.top:rect.height;
  return {passage:active.id,ratio:Math.max(0,Math.min(1,(line-rect.top)/Math.max(1,span))),interface:document.body.dataset.interface||'both'};
}
export function restore(place){
  if(!place||typeof place.passage!=='string'||typeof place.ratio!=='number'||!Number.isFinite(place.ratio)||place.ratio<0||place.ratio>1){announce('Saved reading position is invalid.');return false;}
  if(place.passage==='opening'){scrollTo(0,0);return true;}
  const el=document.getElementById(place.passage);
  if(!el?.matches('[data-passage]')){announce('The saved passage is no longer in this revision.');return false;}
  const passages=[...document.querySelectorAll('main article [data-passage]')],next=passages[passages.indexOf(el)+1];
  const rect=el.getBoundingClientRect(),span=next?next.getBoundingClientRect().top-rect.top:rect.height;
  scrollTo({top:scrollY+rect.top+span*place.ratio-Math.min(130,innerHeight*.2),behavior:'instant'});
  document.body.dataset.interface=place.interface||'both';return true;
}
export function initPlace(){
  const resumed=history.state?.readerPlace;
  if(resumed&&!location.hash)restore(resumed);
  else if(!location.hash){try{const raw=localStorage.getItem(key);if(raw)restore(JSON.parse(raw));}catch(error){storageAvailable=false;announce('Browser storage could not be read. Reading position is unavailable.');console.error(error);}}
  let frame=0, ready=false;
  requestAnimationFrame(()=>requestAnimationFrame(()=>{ready=true;}));
  const progress=document.querySelector('.reading-progress');
  function save(){frame=0;if(document.querySelector('dialog[open]'))return;
    const value=position();const maximum=document.documentElement.scrollHeight-innerHeight;
    if(progress)progress.style.setProperty('--progress',`${maximum>0?Math.max(0,Math.min(100,scrollY/maximum*100)):100}%`);
    if(!ready||!storageAvailable)return;
    try{localStorage.setItem(key,JSON.stringify(value));}catch(error){storageAvailable=false;announce('Browser storage could not be written. Reading position is not being saved.');console.error(error);}
  }
  addEventListener('scroll',()=>{if(!frame)frame=requestAnimationFrame(save);},{passive:true});
  addEventListener('pagehide',save);save();
}
