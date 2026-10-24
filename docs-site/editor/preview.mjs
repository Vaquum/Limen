import MarkdownIt from 'markdown-it';
import {renderPage,escape,content} from '../reader/render.mjs';
import {searchEntries} from '../reader/search-index.mjs';
import {anchorPattern} from '../reader/headings.mjs';

const prefix='/limen/edit/view';
export const viewUrl=(page,pr)=>prefix+page.route+(pr?'?pr='+pr:'');
export function inlinePreview(markdown,page,pages,pr){
  const md=new MarkdownIt({html:false}),env={};md.parse(page.markdown,env);
  md.renderer.rules.link_open=(tokens,index,options,context,renderer)=>{
    const token=tokens[index],href=token.attrGet('href');token.attrSet('data-source-href',href);
    const prefix=content.product.repository+'/blob/main/',local=href.startsWith(prefix)?href.slice(prefix.length):href;
    if(!/^(https?:|mailto:|#)/.test(local)){const target=new URL(local,href.startsWith(prefix)?'https://source.invalid/':'https://source.invalid/'+page.source),mapped=pages.find(item=>item.source===target.pathname.slice(1)||item.url===target.pathname);if(mapped)token.attrSet('href',viewUrl(mapped,pr)+target.hash);}
    return renderer.renderToken(tokens,index,options);
  };
  return md.renderInline(markdown,env);
}
export function referenceIndex(pages){return {source:null,route:'/reference/',url:'/limen/reference/',part:'Reference',title:'Reference',markdown:'# Reference\n\nNames, parameters and boundaries of the two research interfaces.\n\n'+pages.filter(page=>page.part==='Reference').map(page=>'- ['+(page.navigationTitle||page.title)+']('+page.url+')').join('\n')};}
function safeHtml(value){
  if(anchorPattern.test(value.trim()))return value;
  if(/^(?:\s*<!--(?:(?!-->)[\s\S])*-->\s*)+$/.test(value))return value;
  if(/^(?:\s*<\/?details>\s*|\s*<summary>[^<>]*<\/summary>\s*)+$/.test(value))return value;
  return escape(value);
}
export function preview(page,pages,evidence,context){
  let sourceLines=[];
  const mapped=context.passages,authoring={
    configure(md){md.renderer.rules.html_block=(tokens,index)=>{const token=tokens[index],safe=safeHtml(token.content);if(safe!==token.content)return safe;return safe.replace(/<summary>([^<>\n]*)<\/summary>/g,(whole,text,offset)=>{const line=sourceLines[token.map[0]+token.content.slice(0,offset).split('\n').length-1],passage=mapped.find(item=>item.type==='summary'&&item.sourceRange.lines[0]===line&&item.originalText===text);return passage?`<summary data-edit-id="${passage.id}" data-edit-type="summary">${text}</summary>`:whole;});};md.renderer.rules.html_inline=(tokens,index)=>safeHtml(tokens[index].content);},
    annotate(md,kept){sourceLines=kept;md.core.ruler.push('editable_passages',state=>{
      let row=null,column=0;
      for(let index=0;index<state.tokens.length;index++){
        const token=state.tokens[index];if(token.type==='tr_open'){row=token.map;column=0;}if(token.type!=='inline')continue;
        const parent=state.tokens[index-1],table=['td_open','th_open'].includes(parent?.type),map=table?row:parent?.map;
        const cell=table?column++:null;if(!map)continue;
        const passage=mapped.find(item=>item.sourceRange.lines[0]===kept[map[0]]&&item.sourceRange.cell===cell&&item.markdown===token.content);
        if(passage){parent.attrSet('data-edit-id',passage.id);parent.attrSet('data-edit-type',passage.type);}
      }
    });}
  };
  let html=renderPage(page,pages,evidence,{product:'limen-authoring-'+context.state.id,authoring});
  const title=mapped.find(item=>item.type==='title');if(title)html=html.replace('<h1 id=',`<h1 data-edit-id="${title.id}" data-edit-type="title" id=`);
  const introduction=mapped.find(item=>item.type==='paragraph'&&(!title||item.tokenIndex>title.tokenIndex));
  if(introduction)html=html.replace('class="hero-lead"',`class="hero-lead" data-edit-id="${introduction.id}" data-edit-type="paragraph"`);
  const byUrl=new Map(pages.map(item=>[item.url,item]));
  html=html.replace(/(\s)href="([^"<>]+)"/g,(original,spacing,href)=>{const [path,fragment]=href.split('#'),target=byUrl.get(path);return target?`${spacing}href="${viewUrl(target,context.pr)}${fragment?'#'+fragment:''}"`:original;});
  html=html.replaceAll(' data-reference=""','').replaceAll('data-search="/limen/search.json"',`data-search="/limen/search.json${context.pr?'?pr='+context.pr:''}"`);
  if(page.source)html=html.replace(content.product.repository+'/edit/main/'+page.source,content.product.repository+'/blob/'+context.state.head+'/'+page.source).replace('Edit this source</a>','View this source revision</a>');
  html=html.replaceAll('href="/limen/evidence/','href="https://docs.vaquum.fi/limen/evidence/').replaceAll('href="/limen/source/','href="https://docs.vaquum.fi/limen/source/');
  const data={path:page.source,blob:page.blob,head:context.state.head,workspace:context.state.id,route:page.route,pr:context.pr,passages:mapped};
  const banner=`<p class="authoring-identity">Authoring branch <code>${escape(context.state.branch)}</code> · ${escape(context.state.head.slice(0,8))}. Historical proof: Limen ${escape(evidence.package.version)} · ${escape(evidence.revision.head.slice(0,8))}. This branch has not been executed.</p>`;
  return html.replace('<main id="main" tabindex="-1">','<main id="main" tabindex="-1">'+banner).replace('</head>','<link rel="stylesheet" href="/limen/edit/assets/editor.css"><script type="module" src="/limen/edit/assets/preview-client.mjs"></script></head>').replace('</body>',`<template id="editor-source">${escape(JSON.stringify(data))}</template></body>`);
}
export function authoringSearch(pages,pr){return pages.flatMap(page=>searchEntries({...page,url:viewUrl(page,pr)}));}
