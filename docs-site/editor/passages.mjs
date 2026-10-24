import MarkdownIt from 'markdown-it';
import {createHash} from 'node:crypto';
import {headingIdentities,anchorPattern} from '../reader/headings.mjs';

const md=new MarkdownIt({html:true,linkify:false,typographer:false});
const digest=value=>createHash('sha256').update(value).digest('hex');
export class EditError extends Error{constructor(message,status=422){super(message);this.status=status;}}
const requireEdit=(condition,message,status)=>{if(!condition)throw new EditError(message,status);};
function lineOffsets(source){let at=0;return source.split('\n').map(line=>{const offset=at;at+=line.length+1;return offset;});}
function protectedLines(lines){
  const protectedSet=new Set(),stack=[];
  for(let index=0;index<lines.length;index++){
    const start=lines[index].match(/<!--\s*([\w-]+):start\s*-->/),end=lines[index].match(/<!--\s*([\w-]+):end\s*-->/);
    if(start)stack.push(start[1]);
    if(stack.length)protectedSet.add(index);
    if(end){requireEdit(stack.at(-1)===end[1],'Unbalanced protected source region');stack.pop();}
  }
  requireEdit(!stack.length,'Unclosed protected source region');return protectedSet;
}
function cells(line){
  const bounds=[];let start=0;
  for(let index=0;index<line.length;index++)if(line[index]==='|'){
    let escapes=0;for(let at=index-1;at>=0&&line[at]==='\\';at--)escapes++;
    if(escapes%2===0){bounds.push([start,index]);start=index+1;}
  }
  bounds.push([start,line.length]);
  if(!line.slice(...bounds[0]).trim())bounds.shift();if(!line.slice(...bounds.at(-1)).trim())bounds.pop();
  return bounds.map(([start,end])=>{while(/\s/.test(line[start])&&start<end)start++;while(/\s/.test(line[end-1])&&end>start)end--;return [start,end];});
}
export function passages(source,path,baseBlobSha){
  const env={},tokens=md.parse(source,env),lines=source.split('\n'),offsets=lineOffsets(source),protectedSet=protectedLines(lines),result=[];
  let row=null,cell=0;
  for(let index=0;index<tokens.length;index++){
    const token=tokens[index];if(token.type==='tr_open'){row=token.map;cell=0;}
    if(token.type!=='inline'||anchorPattern.test(token.content.trim()))continue;
    const parent=tokens[index-1],table=['th_open','td_open'].includes(parent?.type);
    if(!table&&!['heading_open','paragraph_open'].includes(parent?.type))continue;
    const map=table?row:parent.map;if(!map)continue;
    const column=table?cell++:null;
    if(Array.from({length:map[1]-map[0]},(_,at)=>map[0]+at).some(line=>protectedSet.has(line)))continue;
    let start,end,prefix='';
    if(table){const range=cells(lines[map[0]])[column];requireEdit(range,'Cannot locate table cell');start=offsets[map[0]]+range[0];end=offsets[map[0]]+range[1];}
    else{
      const parts=token.content.split('\n'),first=lines[map[0]].indexOf(parts[0]),lastLine=map[0]+parts.length-1,last=lines[lastLine]?.lastIndexOf(parts.at(-1));
      requireEdit(first>=0&&last>=0,'Cannot locate prose source: '+path+':'+(map[0]+1));
      start=offsets[map[0]]+first;end=offsets[lastLine]+last+parts.at(-1).length;
      prefix=parts.length>1?lines[map[0]+1].slice(0,lines[map[0]+1].indexOf(parts[1])):lines[map[0]].slice(0,first).replace(/(?:[-+*]|\d+[.)])\s/g,match=>' '.repeat(match.length));
    }
    const originalText=source.slice(start,end),sourceRange={start,end,lines:map,cell:column};
    result.push({id:digest(path+':'+baseBlobSha+':'+start+':'+originalText).slice(0,24),path,baseBlobSha,sourceRange,originalText,markdown:token.content,type:table?'cell':parent.tag==='h1'?'title':parent.type==='heading_open'?'heading':'paragraph',tag:parent.tag,prefix,tokenIndex:index,html:md.renderInline(token.content,env)});
  }
  for(let index=0;index<tokens.length;index++)if(tokens[index].type==='html_block'){
    const token=tokens[index];for(const match of token.content.matchAll(/<summary>([^<>\n]*)<\/summary>/g)){
      const line=token.map[0]+token.content.slice(0,match.index).split('\n').length-1;if(protectedSet.has(line))continue;
      const start=offsets[token.map[0]]+match.index+9,end=start+match[1].length,originalText=source.slice(start,end);
      result.push({id:digest(path+':'+baseBlobSha+':'+start+':'+originalText).slice(0,24),path,baseBlobSha,sourceRange:{start,end,lines:[line,line+1],cell:null},originalText,markdown:originalText,type:'summary',tag:'summary',prefix:'',tokenIndex:index});
    }
  }
  return result.sort((a,b)=>a.sourceRange.start-b.sourceRange.start);
}
const signature=tokens=>tokens.filter(token=>!(token.type==='html_block'&&anchorPattern.test(token.content.trim()))).map(token=>[token.type,token.tag,token.nesting,token.hidden,token.type==='fence'?token.info+'\n'+token.content:token.type==='html_block'?token.content.replace(/<summary>[^<>\n]*<\/summary>/g,'<summary></summary>'):null]);
const links=tokens=>tokens.flatMap(token=>(token.children||[]).filter(child=>child.type==='link_open'||child.type==='image').map(child=>[child.type,child.attrGet('href')||child.attrGet('src'),child.attrGet('title')]));
function safeInline(markdown,type){
  requireEdit(typeof markdown==='string'&&markdown.trim()&&Buffer.byteLength(markdown)<=20480,'Passage must contain between 1 byte and 20 KiB');
  requireEdit(!/\n\s*\n/.test(markdown),'New paragraphs are source work');
  requireEdit(!/\]\(\s*<?(?:javascript|data):/i.test(markdown),'Unsafe link target');
  if(type!=='paragraph')requireEdit(!/[\r\n]/.test(markdown),'This passage must stay on one line');
  if(type==='summary')return [];
  const parsed=md.parse(markdown,{});
  requireEdit(parsed.length===3&&parsed[0].type==='paragraph_open','A prose edit cannot introduce block structure');
  requireEdit(!parsed[1].children.some(token=>token.type==='html_inline'||token.type==='image'),'HTML and images are source work');
  return parsed;
}
export function applyPatches(source,path,baseBlobSha,patches){
  requireEdit(Array.isArray(patches)&&patches.length>0&&patches.length<=512,'Commit requires 1–512 passages');
  const mapped=passages(source,path,baseBlobSha),edits=[];
  for(const patch of patches){
    requireEdit(patch.path===path&&patch.baseBlobSha===baseBlobSha,'Source blob moved; refresh and reconcile the draft',409);
    const target=mapped.find(item=>item.sourceRange.start===patch.sourceRange?.start&&item.sourceRange.end===patch.sourceRange?.end);
    requireEdit(target&&target.originalText===patch.originalText,'The original passage no longer matches',409);
    requireEdit(!edits.some(edit=>edit.start<target.sourceRange.end&&edit.end>target.sourceRange.start),'Overlapping prose patches');
    if(patch.newMarkdown===target.markdown){edits.push({start:target.sourceRange.start,end:target.sourceRange.end,text:target.originalText});continue;}
    safeInline(patch.newMarkdown,target.type);
    const env={};md.parse(source,env);const oldInline=md.parseInline(target.markdown,env),newInline=md.parseInline(patch.newMarkdown,env);
    if(target.type!=='summary')requireEdit(JSON.stringify(links(oldInline))===JSON.stringify(links(newInline)),'Adding or retargeting links is source work');
    const replacement=target.type==='summary'?patch.newMarkdown.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;'):patch.newMarkdown;
    const text=replacement.split('\n').map((line,index)=>index?target.prefix+line:line).join('\n');
    edits.push({start:target.sourceRange.start,end:target.sourceRange.end,text});
  }
  let changed=source;for(const edit of edits.sort((a,b)=>b.start-a.start))changed=changed.slice(0,edit.start)+edit.text+changed.slice(edit.end);
  const before=md.parse(source,{}),after=md.parse(changed,{});
  requireEdit(JSON.stringify(signature(before))===JSON.stringify(signature(after)),'Prose edit changed protected content or document structure');
  const oldHeadings=[...headingIdentities(before).values()],newHeadings=[...headingIdentities(after).entries()],offsets=lineOffsets(changed),anchors=[];
  for(let index=0;index<newHeadings.length;index++){
    const [tokenIndex,identity]=newHeadings[index],old=oldHeadings[index];
    if(identity.id!==old.id||identity.kind!==old.kind){requireEdit(!identity.explicit,'An explicit heading identity cannot change');anchors.push({at:offsets[after[tokenIndex].map[0]],text:`<a id="${old.id}" data-reader-heading="${old.kind}"></a>\n\n`});}
  }
  for(const anchor of anchors.sort((a,b)=>b.at-a.at))changed=changed.slice(0,anchor.at)+anchor.text+changed.slice(anchor.at);
  requireEdit(new Set([...headingIdentities(md.parse(changed,{})).values()].map(item=>item.id)).size===oldHeadings.length,'Heading rename makes a destination ambiguous');
  return changed;
}
