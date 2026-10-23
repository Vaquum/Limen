import MarkdownIt from 'markdown-it';
import {pageLabel} from './render.mjs';

import {headingIdentities} from './headings.mjs';

// Index the same parsed headings and literal code that readers can look up.
export function searchEntries(page){
  const tokens=new MarkdownIt({html:true}).parse(page.markdown,{}),identities=headingIdentities(new MarkdownIt({html:true}).parse(page.markdown,{})),entries=[];
  let entry={title:pageLabel(page),heading:'',part:page.part,route:page.url,text:''};
  const plain=token=>(token.children||[]).map(child=>['text','code_inline','image'].includes(child.type)?child.content:['softbreak','hardbreak'].includes(child.type)?' ':'').join('');
  for(let index=0;index<tokens.length;index++){
    const token=tokens[index];
    if(token.type==='heading_open'){
      const heading=plain(tokens[index+1]),id=identities.get(index).id;
      if(token.tag!=='h1'){
        entries.push(entry);
        entry={title:pageLabel(page),heading,part:page.part,route:page.url+'#'+id,text:''};
      }
    }
    if(token.type==='inline')entry.text+=plain(token)+' ';
    if(token.type==='fence'||token.type==='code_block')entry.text+=token.content+' ';
  }
  entries.push(entry);
  return entries.map(item=>({...item,text:item.text.replace(/\s+/g,' ').trim()})).filter(item=>item.text);
}
