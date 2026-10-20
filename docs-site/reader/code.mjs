export function initCode(root=document) {
  for(const figure of root.querySelectorAll('.code')) {
    if(figure.dataset.enhanced)continue;
    figure.dataset.enhanced='true';
    for(const button of figure.querySelectorAll('button'))button.hidden=false;
    figure.querySelector('[data-copy]').addEventListener('click',async event=>{
      const button=event.currentTarget;
      const bytes=Uint8Array.from(atob(figure.dataset.source),c=>c.charCodeAt(0));
      const text=new TextDecoder().decode(bytes);
      try{await navigator.clipboard.writeText(text);button.textContent='Copied';}
      catch(error){document.querySelector('[data-status]').textContent='Code could not be copied. Select the source text to copy it.';console.error(error);}
    });
    figure.querySelector('[data-wrap]').addEventListener('click',event=>{
      const off=figure.classList.toggle('nowrap');event.currentTarget.setAttribute('aria-pressed',String(!off));
    });
  }
}
