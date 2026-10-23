import MarkdownIt from 'markdown-it';
import {renderPage,escape,content} from '../reader/render.mjs';
import {searchEntries} from '../reader/search-index.mjs';
import {anchorPattern} from '../reader/headings.mjs';

const prefix='/limen/edit/view';
export const viewUrl=(page,pr)=>prefix+page.route+(pr?'?pr='+pr:'');
export function inlinePreview(markdown,source){const md=new MarkdownIt({html:false}),env={};md.parse(source,env);return md.renderInline(markdown,env);}
export function referenceIndex(pages){return {source:null,route:'/reference/',url:'/limen/reference/',part:'Reference',title:'Reference',markdown:'# Reference\n\nNames, parameters and boundaries of the two research interfaces.\n\n'+pages.filter(page=>page.part==='Reference').map(page=>'- ['+(page.navigationTitle||page.title)+']('+page.url+')').join('\n')};}
function safeHtml(value){
  if(anchorPattern.test(value.trim()))return value;
  if(/^(?:\s*<!--(?:(?!-->)[\s\S])*-->\s*)+$/.test(value))return value;
  if(/^(?:\s*<\/?details>\s*|\s*<summary>[^<>]*<\/summary>\s*)+$/.test(value))return value;
  return escape(value);
}
export function preview(page,pages,evidence,context){
  const mapped=context.passages,authoring={
    configure(md){md.renderer.rules.html_block=(tokens,index)=>safeHtml(tokens[index].content);md.renderer.rules.html_inline=(tokens,index)=>safeHtml(tokens[index].content);},
    annotate(md,kept){md.core.ruler.push('editable_passages',state=>{
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
  html=html.replace(/href="([^"<>]+)"/g,(original,href)=>{const [path,fragment]=href.split('#'),target=byUrl.get(path);return target?`href="${viewUrl(target,context.pr)}${fragment?'#'+fragment:''}"`:original;});
  html=html.replaceAll(' data-reference=""','').replaceAll('data-search="/limen/search.json"',`data-search="/limen/search.json${context.pr?'?pr='+context.pr:''}"`);
  html=html.replaceAll('href="/limen/evidence/','href="https://docs.vaquum.fi/limen/evidence/').replaceAll('href="/limen/source/','href="https://docs.vaquum.fi/limen/source/');
  const data={path:page.source,blob:page.blob,head:context.state.head,workspace:context.state.id,route:page.route,pr:context.pr,passages:mapped};
  const banner=`<p class="authoring-identity">Authoring branch <code>${escape(context.state.branch)}</code> · ${escape(context.state.head.slice(0,8))}. Historical proof: Limen ${escape(evidence.package.version)} · ${escape(evidence.revision.head.slice(0,8))}. This branch has not been executed.</p>`;
  return html.replace('<main id="main" tabindex="-1">','<main id="main" tabindex="-1">'+banner).replace('</head>','<link rel="stylesheet" href="/limen/edit/assets/editor.css"><script type="module" src="/limen/edit/assets/preview-client.mjs"></script></head>').replace('</body>',`<template id="editor-source">${escape(JSON.stringify(data))}</template></body>`);
}
export function authoringSearch(pages,pr){return pages.flatMap(page=>searchEntries({...page,url:viewUrl(page,pr)}));}
