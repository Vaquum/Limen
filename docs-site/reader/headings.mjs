import MarkdownIt from 'markdown-it';
export function slug(text){return text.replace(/[^\p{L}\p{N}_\- ]/gu,'').trim().toLowerCase().replaceAll(' ','-');}
export const anchorPattern=/^<a id="([\p{L}\p{N}_-]+)" data-reader-heading="(python|compiled|plain)"><\/a>\s*$/u;
export function headingIdentities(tokens){
  const seen=new Map(),identities=new Map();
  for(let index=0;index<tokens.length;index++){
    const token=tokens[index];if(token.type!=='heading_open')continue;
    const text=tokens[index+1].content.replaceAll('`',''),stem=slug(token.tag==='h1'?plainHeading(tokens[index+1]):text),count=seen.get(stem)||0;
    if(token.tag!=='h1')seen.set(stem,count+1);
    const before=tokens[index-1]?.type==='html_block'?tokens[index-1]:tokens[index-1]?.type==='paragraph_close'&&tokens[index-2]?.type==='inline'?tokens[index-2]:null;
    const explicit=before?.content.trim().match(anchorPattern);
    identities.set(index,{id:explicit?.[1]||(count?`${stem}-${count}`:stem),kind:explicit?.[2]||(/\bCompiled\b/.test(text)?'compiled':/\bPython\b/.test(text)?'python':'plain'),explicit:!!explicit});
  }
  return identities;
}

export const plainHeading=inline=>(inline.children||[]).map(child=>['text','code_inline','image'].includes(child.type)?child.content:['softbreak','hardbreak'].includes(child.type)?' ':'').join('');
export function documentTitle(markdown){const tokens=new MarkdownIt({html:true}).parse(markdown,{}),at=tokens.findIndex(token=>token.type==='heading_open'&&token.tag==='h1');return at<0?null:plainHeading(tokens[at+1]);}
