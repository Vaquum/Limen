import MarkdownIt from 'markdown-it';
import {slug,pageLabel} from './render.mjs';

// Index the same parsed headings and literal code that readers can look up.
export function searchEntries(page){
  const tokens=new MarkdownIt({html:true}).parse(page.markdown,{}),seen=new Map(),entries=[];
  let entry={title:pageLabel(page),heading:'',part:page.part,route:page.url,text:''};
  const plain=token=>(token.children||[]).map(child=>['text','code_inline','image'].includes(child.type)?child.content:['softbreak','hardbreak'].includes(child.type)?' ':'').join('');
  for(let index=0;index<tokens.length;index++){
    const token=tokens[index];
    if(token.type==='heading_open'){
      const heading=plain(tokens[index+1]),stem=slug(heading),count=seen.get(stem)||0;
      if(token.tag!=='h1'){
        seen.set(stem,count+1);
        entries.push(entry);
        entry={title:pageLabel(page),heading,part:page.part,route:page.url+'#'+stem+(count?'-'+count:''),text:''};
      }
    }
    if(token.type==='inline')entry.text+=plain(token)+' ';
    if(token.type==='fence'||token.type==='code_block')entry.text+=token.content+' ';
  }
  entries.push(entry);
  return entries.map(item=>({...item,text:item.text.replace(/\s+/g,' ').trim()})).filter(item=>item.text);
}
