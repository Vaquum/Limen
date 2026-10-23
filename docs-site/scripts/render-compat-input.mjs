import MarkdownIt from 'markdown-it';
import {anchorPattern} from '../reader/headings.mjs';
const md=new MarkdownIt({html:true});
export function compatibilityMarkdown(source){
  const tokens=md.parse(source,{}),removed=new Set();
  for(let index=0;index<tokens.length;index++){
    const token=tokens[index],map=token.type==='html_block'?token.map:token.type==='inline'&&tokens[index-1]?.type==='paragraph_open'?tokens[index-1].map:null;
    if(map&&anchorPattern.test(token.content.trim()))for(let line=map[0];line<map[1];line++)removed.add(line);
  }
  return source.split('\n').filter((_,index)=>!removed.has(index)).join('\n');
}
