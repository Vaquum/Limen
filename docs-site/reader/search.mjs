import {announce} from './place.mjs';
export function initSearch(){
  const button=document.querySelector('[data-search-open]');const dialog=document.querySelector('[data-search-dialog]');
  const input=dialog.querySelector('input');const results=dialog.querySelector('ol');let index=null,opener=null;
  async function open(){opener=document.activeElement;dialog.showModal();input.focus();
    if(!index){input.disabled=true;dialog.querySelector('[data-search-count]').textContent='Loading search…';try{const response=await fetch(document.body.dataset.search);if(!response.ok)throw Error(`Search index returned ${response.status}`);index=await response.json();input.disabled=false;input.focus();dialog.querySelector('[data-search-count]').textContent='Type at least two letters.';}
      catch(error){dialog.querySelector('[data-search-count]').textContent='Search is unavailable. Use the chapter and reference links.';announce('Search is unavailable.');console.error(error);}}
  }
  button.hidden=false;button.addEventListener('click',open);
  dialog.querySelector('[data-search-close]').addEventListener('click',()=>dialog.close());dialog.addEventListener('close',()=>opener?.focus());
  dialog.addEventListener('keydown',event=>{if(event.key==='Escape'){event.preventDefault();dialog.close();}});
  input.addEventListener('input',()=>{results.replaceChildren();if(!index)return;const query=input.value.trim().toLowerCase();if(query.length<2)return;
    const found=index.map(page=>({...page,at:page.text.toLowerCase().indexOf(query)})).filter(page=>page.at>=0||page.title.toLowerCase().includes(query)).sort((a,b)=>Number(b.title.toLowerCase().includes(query))-Number(a.title.toLowerCase().includes(query))).slice(0,20);
    for(const page of found){const li=document.createElement('li'),a=document.createElement('a'),p=document.createElement('p');a.href=page.route;a.textContent=page.title;p.textContent=page.text.slice(Math.max(0,page.at-48),Math.max(0,page.at-48)+180);li.append(a,p);results.append(li);}
    dialog.querySelector('[data-search-count]').textContent=`${found.length} results`;
  });
  document.addEventListener('keydown',event=>{const editing=event.target.closest('input,textarea,select,[contenteditable]');if(!editing&&((event.key.toLowerCase()==='k'&&(event.metaKey||event.ctrlKey))||event.key==='/')){event.preventDefault();open();}});
}
