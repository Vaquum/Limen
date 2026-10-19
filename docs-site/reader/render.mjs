import MarkdownIt from 'markdown-it';
import Prism from 'prismjs';
import 'prismjs/components/prism-python.js';
import 'prismjs/components/prism-yaml.js';
import 'prismjs/components/prism-bash.js';
import 'prismjs/components/prism-json.js';
import {createHash} from 'node:crypto';
export const hash=value=>createHash('sha256').update(value).digest('hex');
export const escape=value=>String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
export function slug(text){return text.replace(/[^\p{L}\p{N}_\- ]/gu,'').trim().toLowerCase().replaceAll(' ','-');}
export function renderMarkdown(page,pages,evidence){
  const md=new MarkdownIt({html:true,linkify:false,typographer:false});const headings=[],seen=new Map();let codeCount=0,linkCount=0;
  md.renderer.rules.heading_open=(tokens,index,options,env,renderer)=>{const token=tokens[index],text=tokens[index+1].content.replaceAll('`','');const stem=slug(text),count=seen.get(stem)||0;seen.set(stem,count+1);const id=count?`${stem}-${count}`:stem;token.attrSet('id',id);token.attrSet('tabindex','-1');if(token.tag==='h2')headings.push({title:text,id});return renderer.renderToken(tokens,index,options);};
  md.renderer.rules.paragraph_open=(tokens,index,options,env,renderer)=>{const text=tokens[index+1]?.content||'';const id=`p-${hash(text).slice(0,12)}`;const count=seen.get(id)||0;seen.set(id,count+1);tokens[index].attrSet('id',count?`${id}-${count}`:id);tokens[index].attrSet('data-passage','');return renderer.renderToken(tokens,index,options);};
  const originalLink=md.renderer.rules.link_open||((tokens,index,options,env,renderer)=>renderer.renderToken(tokens,index,options));
  md.renderer.rules.link_open=(tokens,index,options,env,renderer)=>{const token=tokens[index],href=token.attrGet('href');let target=null,anchor='';
    if(href&&!/^(https?:|mailto:|#)/.test(href)){const split=href.split('#');anchor=split[1]||'';const resolved=new URL(split[0],`https://source.invalid/${page.source}`).pathname.slice(1);target=pages.find(p=>p.source===resolved);}
    if(target){token.attrSet('href',target.url+(anchor?`#${anchor}`:''));token.attrSet('id',`link-${hash(page.source+':'+href+':'+linkCount++).slice(0,12)}`);if(page.part==='Book'&&target.part==='Reference')token.attrSet('data-reference','');}
    else if(href&&!/^(https?:|mailto:|#|\/)/.test(href))token.attrSet('href',`https://github.com/Vaquum/Limen/blob/${evidence.revision.head}/${new URL(href,`https://source.invalid/${page.source}`).pathname.slice(1)}`);
    return originalLink(tokens,index,options,env,renderer);
  };
  md.renderer.rules.fence=(tokens,index)=>{const token=tokens[index],language=token.info.trim().split(/\s+/)[0]||'text',source=token.content;const id=`code-${++codeCount}`;const grammar=Prism.languages[language==='python-fragment'?'python':language==='sh'?'bash':language];
    const renderedLines=[''];
    function append(value,classes=[]){
      if(typeof value==='string'){const parts=value.split('\n');for(let n=0;n<parts.length;n++){if(n)renderedLines.push('');const text=escape(parts[n]);renderedLines[renderedLines.length-1]+=classes.length?`<span class="token ${classes.map(escape).join(' ')}">${text}</span>`:text;}}
      else if(Array.isArray(value))value.forEach(token=>append(token,classes));
      else append(value.content,[...classes,value.type]);
    }
    append(grammar?Prism.tokenize(source.replace(/\n$/,''),grammar):source.replace(/\n$/,''));
    const lines=renderedLines.map((line,i)=>`<span class="code-line" id="${id}-L${i+1}"><a class="line-number" href="#${id}-L${i+1}" aria-hidden="true" tabindex="-1">${i+1}</a><span class="code-src">${line||' '}</span></span>`).join('');
    const before=tokens[index-1];const marker=before?.type==='html_block'?before.content.match(/<!-- example: (\S+) source=(\S+) region=(\S+) scenario=(\S+) interface=(\S+) role=executed -->/):null;
    let record=null;if(marker){record=evidence.sources.find(s=>s.path===marker[2]&&s.region===marker[3]);if(!record||record.sha256!==hash(source))throw Error(`Missing or changed executable binding: ${marker[1]}`);}
    const sourcePath=record?.path||page.source,commit=record?.commit||evidence.revision.head;
    const identity=record?.status==='merged'?`<a href="${escape(record.merged_pr)}">Merged source PR</a>`:'Candidate source · unmerged';
    const validated=marker?evidence.validation.scenarios.find(s=>s.id===marker[4]&&s.interface===marker[5]&&s.status==='passed'):null;
    if(marker&&!validated)throw Error(`No successful execution for ${marker[1]}`);
    return `<figure class="code" id="${id}" data-source="${Buffer.from(source).toString('base64')}"><figcaption class="code-bar"><span>${escape(language)}${record?' · '+escape(record.path.split('/').at(-1)):''}</span><span><button type="button" data-wrap aria-pressed="true" hidden>Wrap</button> <button type="button" data-copy hidden>Copy</button></span></figcaption><pre><code>${lines}</code></pre><div class="source">${identity} · <a href="https://github.com/Vaquum/Limen/blob/${commit}/${escape(sourcePath)}">${escape(commit.slice(0,8))}</a>${validated?`<span class="claim">Executed with Limen ${escape(evidence.package.version)} · <a href="/limen/evidence/current.json">validation, inputs and tests</a> · <a href="/limen/source/${escape(sourcePath)}">complete source</a> · <a href="/limen/evidence/${escape(evidence.bundle.path.split('/').at(-1))}">project and recorded inputs</a></span>`:'<span class="claim">Reference excerpt · execution is not asserted by this block.</span>'}</div></figure>`;
  };
  const originalTableOpen=md.renderer.rules.table_open||(()=>'<table>');const originalTableClose=md.renderer.rules.table_close||(()=>'</table>');
  md.renderer.rules.table_open=(...args)=>'<div class="table-scroll" tabindex="0" role="region" aria-label="Scrollable reference table">'+originalTableOpen(...args);
  md.renderer.rules.table_close=(...args)=>originalTableClose(...args)+'</div>';
  let body=page.markdown.replace(/^# .+\n/, '');const introduction=body.match(/^\s*([^#<\n][^\n]+)\n\n/);if(introduction)body=body.replace(introduction[0], '');return {html:md.render(body),headings};
}
export function renderPage(page,pages,evidence,{product='limen'}={}){
  const {html,headings}=renderMarkdown(page,pages,evidence);const book=pages.filter(p=>p.bookOrder!==undefined).sort((a,b)=>a.bookOrder-b.bookOrder);const at=book.indexOf(page),next=at>=0?book[at+1]:null,previous=at>0?book[at-1]:null;
  const home=book[0];const reference=pages.find(p=>p.route==='/reference/'),developer=pages.find(p=>p.source==='docs/Developer/README.md');
  const navigation=page.part==='Book'?book:pages.filter(p=>p.part===page.part&&p.source).sort((a,b)=>a.title.localeCompare(b.title));
  const nav=navigation.map(p=>`<li><a href="${p.url}"${p===page?' aria-current="page"':''}>${escape(p.title)}</a></li>`).join('');
  const contents=headings.map(h=>`<li><a href="#${escape(h.id)}">${escape(h.title)}</a></li>`).join('');
  const firstParagraph=page.markdown.match(/\n\n([^#<\n][^\n]+)/)?.[1]||'Limen research, interfaces and evidence.';
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${escape(page.title)} — Limen</title><meta name="description" content="${escape(firstParagraph)}"><link rel="canonical" href="https://docs.vaquum.fi${page.url}"><link rel="stylesheet" href="/limen/assets/book.css"><script type="module" src="/limen/assets/main.mjs"></script></head><body data-product="${product}" data-content="${escape(page.source||'index:'+page.part)}" data-interface="both" data-search="/limen/search.json"><a class="skip" href="#main">Skip to the chapter</a><header class="site-header"><div class="header-inner"><a class="wordmark" href="${home.url}">LIMEN</a><nav class="part-nav" aria-label="Parts"><a href="${home.url}"${page.part==='Book'?' aria-current="true"':''}>Book</a><a href="${reference.url}"${page.part==='Reference'?' aria-current="true"':''}>Reference</a><a href="${developer.url}"${page.part==='Developer'?' aria-current="true"':''}>Developer</a><button type="button" data-search-open hidden>Search</button></nav></div><div class="reading-progress"></div></header><div class="hero"><div class="hero-inner"><p class="label">${escape(page.part)} / Limen research</p><h1>${escape(page.title)}</h1><p class="hero-lead">${escape(firstParagraph)}</p></div></div><div class="layout"><aside class="chapter-nav" aria-label="Book chapters"><p class="label">${page.part==='Book'?'The research book':escape(page.part)}</p><ul>${nav}</ul><nav class="chapter-contents" aria-label="In this chapter"><p class="label">In this chapter</p><ol>${contents}</ol></nav></aside><main id="main" tabindex="-1"><article>${html}</article><p class="status" data-status role="status" aria-live="polite"></p><div class="page-end"><span>${previous?`Previous<a href="${previous.url}">${escape(previous.title)}</a>`:'Limen / Python Research and Compiled Research'}</span><span>${next?`Continue<a href="${next.url}">${escape(next.title)}</a>`:`<a href="${home.url}">Return to the research book</a>`}</span></div><div class="page-end">${page.source?`<a href="https://github.com/Vaquum/Limen/edit/main/${escape(page.source)}">Edit this source</a>`:'Generated from the canonical source map'}<span>Candidate reader · ${escape(evidence.revision.head.slice(0,8))}</span></div></main></div><dialog class="reference-dialog" data-reference-dialog aria-label="Reference"><header class="dialog-head"><span class="label">Reference / keep your place</span><button type="button" data-reference-close>Return to chapter</button></header><div class="return-rule"></div><div class="reference-body" data-reference-body></div></dialog><dialog class="search-dialog" data-search-dialog aria-label="Search Limen"><header class="dialog-head"><span class="label">Find a passage</span><button type="button" data-search-close>Close</button></header><div class="search-body"><label for="search">Search the book and references</label><input id="search" type="search" autocomplete="off"><p data-search-count class="status" aria-live="polite"></p><ol></ol></div></dialog></body></html>`;
}
