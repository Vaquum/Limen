import {announce} from './place.mjs';
export function initSearch(){
  const button=document.querySelector('[data-search-open]');const dialog=document.querySelector('[data-search-dialog]');
  const input=dialog.querySelector('input');const results=dialog.querySelector('ol');let index=null,opener=null;
  async function open(){opener=document.activeElement;dialog.showModal();input.focus();
    if(!index){input.disabled=true;dialog.querySelector('[data-search-count]').textContent='Loading search…';try{const response=await fetch(document.body.dataset.search);if(!response.ok)throw Error(`Search index returned ${response.status}`);index=await response.json();input.disabled=false;input.focus();dialog.querySelector('[data-search-count]').textContent='Type at least two letters.';}
      catch(error){dialog.querySelector('[data-search-count]').textContent='Search is unavailable. Use the chapter and reference links.';announce('Search is unavailable.');console.error(error);}}
  }
  button.hidden=false;button.addEventListener('click',open);
  dialog.querySelector('[data-search-close]').addEventListener('click',()=>dialog.close());dialog.addEventListener('close',()=>opener?.focus({preventScroll:true}));
  results.addEventListener('click',event=>{if(event.target.closest('a[href]')&&event.button===0&&!event.metaKey&&!event.ctrlKey&&!event.shiftKey&&!event.altKey)dialog.close();});
  dialog.addEventListener('keydown',event=>{if(event.key==='Escape'){event.preventDefault();dialog.close();}});
  input.addEventListener('input',()=>{results.replaceChildren();if(!index)return;const query=input.value.trim().toLowerCase();if(query.length<2){dialog.querySelector('[data-search-count]').textContent='Type at least two letters.';return;}
    const terms=query.split(/\s+/);
    const matches=index.map(page=>{const label=(page.title+' '+(page.heading||'')).toLowerCase(),text=page.text.toLowerCase();return {...page,at:text.indexOf(terms[0]),score:terms.reduce((sum,term)=>sum+(label.includes(term)?4:0),0),matches:terms.every(term=>label.includes(term)||text.includes(term))};}).filter(page=>page.matches).sort((a,b)=>b.score-a.score||a.title.localeCompare(b.title));
    for(const page of matches.slice(0,20)){const li=document.createElement('li'),a=document.createElement('a'),p=document.createElement('p'),context=document.createElement('small');a.href=page.route;a.textContent=page.title+(page.heading?' / '+page.heading:'');context.textContent=page.part;p.textContent=page.text.slice(Math.max(0,page.at-48),Math.max(0,page.at-48)+180);li.append(context,a,p);results.append(li);}
    dialog.querySelector('[data-search-count]').textContent=matches.length>20?`Showing 20 of ${matches.length} passages`:`${matches.length} results`;
  });
  document.addEventListener('keydown',event=>{const editing=event.target.closest('input,textarea,select,[contenteditable]');if(!editing&&((event.key.toLowerCase()==='k'&&(event.metaKey||event.ctrlKey))||event.key==='/')){event.preventDefault();open();}});
}
