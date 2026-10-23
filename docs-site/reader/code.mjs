export function initCode(root=document) {
  for(const figure of root.querySelectorAll('.code')) {
    if(figure.dataset.enhanced)continue;
    figure.dataset.enhanced='true';
    for(const button of figure.querySelectorAll('button'))button.hidden=false;
    figure.querySelector('[data-copy]').addEventListener('click',async event=>{
      const button=event.currentTarget;
      const bytes=Uint8Array.from(atob(figure.dataset.source),c=>c.charCodeAt(0));
      const text=new TextDecoder().decode(bytes);
      try{await navigator.clipboard.writeText(text);button.textContent='Copied';setTimeout(()=>{button.textContent='Copy';},1800);}
      catch(error){document.querySelector('[data-status]').textContent='Code could not be copied. Select the source text to copy it.';console.error(error);}
    });
    figure.querySelector('[data-wrap]').addEventListener('click',event=>{
      const off=figure.classList.toggle('nowrap');event.currentTarget.setAttribute('aria-pressed',String(!off));
    });
    if(figure.querySelectorAll('.code-line').length>18){
      const expand=document.createElement('button');expand.type='button';expand.textContent='Expand';expand.setAttribute('aria-expanded','false');
      figure.querySelector('.code-bar').lastElementChild.prepend(expand);
      expand.addEventListener('click',()=>{const full=figure.classList.toggle('expanded');expand.setAttribute('aria-expanded',String(full));expand.textContent=full?'Collapse':'Expand';});
    }
    const numbers=[...figure.querySelectorAll('.line-number')];
    if(numbers.length)numbers[0].tabIndex=0;
    let start=0;
    function selectLine(index,extend){
      if(!extend)start=index;
      const low=Math.min(start,index),high=Math.max(start,index);
      numbers.forEach((number,at)=>{number.tabIndex=at===index?0:-1;number.parentElement.classList.toggle('selected-line',at>=low&&at<=high);});
      history.replaceState(history.state,'',numbers[low].getAttribute('href')+(low===high?'':'-L'+(high+1)));
    }
    numbers.forEach((number,index)=>{number.removeAttribute('aria-hidden');number.setAttribute('aria-label','Line '+(index+1));
      number.addEventListener('click',event=>{if(event.button!==0||event.metaKey||event.ctrlKey||event.altKey)return;event.preventDefault();selectLine(index,event.shiftKey);});
      number.addEventListener('keydown',event=>{if(event.key==='ArrowDown'||event.key==='ArrowUp'){
        event.preventDefault();const next=Math.max(0,Math.min(numbers.length-1,index+(event.key==='ArrowDown'?1:-1)));selectLine(next,event.shiftKey);numbers[next].focus();
      }});
    });
    function readRange(event){
      numbers.forEach(number=>number.parentElement.classList.remove('selected-line'));
      const range=location.hash.match(new RegExp('^#'+figure.id.replace(/^reference-/,'')+'-L(\\d+)(?:-L(\\d+))?$'));
      if(range){const low=Number(range[1])-1,high=Number(range[2]||range[1])-1;
        if(low>=0&&high>=low&&high<numbers.length){start=low;selectLine(high,true);if(event?.detail?.scroll!==false)numbers[low].parentElement.scrollIntoView({block:'center'});}
      }
    }
    figure.addEventListener('reader-code-range',readRange);readRange();
  }
}
export function revealCodeRange({scroll=true}={}){for(const figure of document.querySelectorAll('.code[data-enhanced]'))figure.dispatchEvent(new CustomEvent('reader-code-range',{detail:{scroll}}));}
