import MarkdownIt from 'markdown-it';
import {anchorPattern,headingIdentities,plainHeading} from '../reader/headings.mjs';
import {escape} from '../reader/render.mjs';
const md=new MarkdownIt({html:true});
export function compatibilityMarkdown(source){
  const tokens=md.parse(source,{}),removed=new Set();
  for(let index=0;index<tokens.length;index++){
    const token=tokens[index],map=token.type==='html_block'?token.map:token.type==='inline'&&tokens[index-1]?.type==='paragraph_open'?tokens[index-1].map:null;
    if(map&&anchorPattern.test(token.content.trim()))for(let line=map[0];line<map[1];line++)removed.add(line);
  }
  return source.split('\n').filter((_,index)=>!removed.has(index)).join('\n');
}
export function compatibilityContents(html,source){
  const tokens=md.parse(source,{});
  for(const [index,identity] of headingIdentities(tokens)){
    if(tokens[index].tag==='h1')continue;const inline=tokens[index+1],raw=inline.content.replaceAll('`','');
    html=html.replaceAll(`href="#${escape(identity.id)}">${escape(raw)}</a>`,`href="#${escape(identity.id)}">${escape(plainHeading(inline))}</a>`);
  }
  return html;
}
