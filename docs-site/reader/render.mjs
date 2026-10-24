import MarkdownIt from 'markdown-it';
import Prism from 'prismjs';
import 'prismjs/components/prism-python.js';
import 'prismjs/components/prism-yaml.js';
import 'prismjs/components/prism-bash.js';
import 'prismjs/components/prism-json.js';
import {createHash} from 'node:crypto';
import CONTENT from './content.json' with {type:'json'};
import {slug,headingIdentities,anchorPattern,plainHeading} from './headings.mjs';
export const content=CONTENT;
export {slug} from './headings.mjs';
export const hash=value=>createHash('sha256').update(value).digest('hex');
export const escape=value=>String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
export const pageLabel=page=>page.navigationTitle||page.title;
function documentText(markdown,{omitPreamble=false}={}){
  const tokens=new MarkdownIt({html:true}).parse(markdown,{}),lines=markdown.split('\n');
  const at=tokens.findIndex(token=>token.type==='heading_open'&&token.tag==='h1');
  const removed=[];if(at>=0){removed.push(tokens[at].map);const before=tokens[at-1]?.type==='html_block'?tokens[at-1]:tokens[at-1]?.type==='paragraph_close'?tokens[at-2]:null;if(before&&anchorPattern.test(before.content.trim()))removed.push(before.map);if(omitPreamble)removed.push([0,tokens[at].map[0]]);}
  let first=at>=0?at+3:0;
  while(true){if(tokens[first]?.type==='html_block'&&(tokens[first].content.trim().startsWith('<!--')||anchorPattern.test(tokens[first].content.trim()))){first++;continue;}if(tokens[first]?.type==='paragraph_open'&&anchorPattern.test(tokens[first+1]?.content.trim()||'')){first+=3;continue;}break;}
  if(tokens[first]?.type==='blockquote_open')first++;
  const paragraph=tokens[first]?.type==='paragraph_open'?tokens[first]:null;
  const inline=paragraph?tokens[first+1]:null;
  const introduction=inline?.children?.map(token=>['text','code_inline','image'].includes(token.type)?token.content:['softbreak','hardbreak'].includes(token.type)?' ':'').join('')||'';
  if(paragraph)removed.push(paragraph.map);
  const kept=lines.map((_,line)=>line).filter(line=>!removed.some(range=>range&&line>=range[0]&&line<range[1]));
  return {body:kept.map(line=>lines[line]).join('\n'),kept,introduction,introductionSource:inline?.content||''};
}
export function renderMarkdown(page,pages,evidence,results={},authoring){
  const md=new MarkdownIt({html:true,linkify:false,typographer:false});const headings=[],seen=new Map();let codeCount=0,linkCount=0,identities=new Map();
  md.core.ruler.push('reader_heading_identities',state=>{identities=headingIdentities(state.tokens);});
  const htmlBlock=md.renderer.rules.html_block;
  if(authoring)authoring.configure(md);
  const previewHtml=md.renderer.rules.html_block;
  md.renderer.rules.html_block=(tokens,index,options,env,renderer)=>{
    const match=tokens[index].content.trim().match(/^<!-- result: (\S+) interface=(python|compiled) scenario=(\S+) fields=([\w.\[\],]+) -->$/);
    if(!match){if(anchorPattern.test(tokens[index].content.trim()))return '';return (authoring?previewHtml:htmlBlock)(tokens,index,options,env,renderer);}
    if(authoring)return `<figure class="result" data-research-interface="${match[2]}" id="${escape(match[1])}"><figcaption>${escape(match[1])}</figcaption><p>Published result unavailable in this authoring preview. This branch has not been executed.</p></figure>`;
    const result=results[match[2]+'/'+match[3]];
    if(!result)throw Error('Missing current result: '+match[1]);
    function select(value,parts){if(!parts.length)return value;const [part,...tail]=parts,key=part.endsWith('[]')?part.slice(0,-2):part;if(value?.[key]===undefined)throw Error('Missing result field: '+parts.join('.'));if(part.endsWith('[]')){if(!Array.isArray(value[key]))throw Error('Result field is not a row collection: '+key);return value[key].map(row=>select(row,tail));}return select(value[key],tail);}
    const fields=match[4].split(','),selected=fields.map(field=>select(result.value,field.split('.')));
    const rowPrefix=fields[0].includes('[].')?fields[0].split('[].')[0]:null;
    const table=rowPrefix&&fields.every(field=>field.startsWith(rowPrefix+'[].'));
    const label=field=>escape(field.replaceAll('_',' ').replaceAll('[]',''));
    const rows=table?`<div class="table-scroll" tabindex="0" role="region" aria-label="Recorded result table"><table><thead><tr>${fields.map(field=>`<th scope="col">${label(field.split('[].')[1])}</th>`).join('')}</tr></thead><tbody>${selected[0].map((_,index)=>`<tr>${selected.map(column=>`<td>${escape(JSON.stringify(column[index]))}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`:`<dl>${fields.map((field,index)=>`<dt>${label(field)}</dt><dd><pre>${escape(JSON.stringify(selected[index],null,2))}</pre></dd>`).join('')}</dl>`;
    const source=evidence.sources.find(item=>item.path===result.source&&item.region==='__file__');
    if(!source)throw Error('Result has no source identity: '+match[1]);
    return `<figure class="result" data-research-interface="${match[2]}" id="${escape(match[1])}"><figcaption>${match[2]==='python'?'Python':'Compiled'} Research · ${escape(match[3])} · current execution</figcaption>${rows}<p class="status">${escape(content.product.name)} ${escape(evidence.package.version)} · <a href="${content.product.repository}/blob/${source.commit}/${escape(source.path)}">${source.commit.slice(0,8)}</a>${source.merged_pr?` · <a href="${escape(source.merged_pr)}">Merged source PR</a>`:' · candidate source'} · <a href="${content.product.basePath}/evidence/current.json">Validation and result digest</a> · <a href="${content.product.basePath}/evidence/${escape(evidence.native_outputs.path.split('/').at(-1))}">Native records</a></p></figure>`;
  };
  md.renderer.rules.heading_open=(tokens,index,options,env,renderer)=>{const token=tokens[index],text=plainHeading(tokens[index+1]);const identity=identities.get(index),id=identity.id;token.attrSet('id',id);if(identity.explicit)token.attrSet('data-heading-interface',identity.kind);token.attrSet('tabindex','-1');if(token.tag==='h2')headings.push({title:text,id});return renderer.renderToken(tokens,index,options);};
  md.renderer.rules.paragraph_open=(tokens,index,options,env,renderer)=>{const text=tokens[index+1]?.content||'';if(anchorPattern.test(text.trim())){tokens[index+1].children=[];return '';}const id=`p-${hash(text).slice(0,12)}`;const count=seen.get(id)||0;seen.set(id,count+1);tokens[index].attrSet('id',count?`${id}-${count}`:id);tokens[index].attrSet('data-passage','');return renderer.renderToken(tokens,index,options);};
  md.renderer.rules.paragraph_close=(tokens,index,options,env,renderer)=>anchorPattern.test(tokens[index-1]?.content?.trim()||'')?'':renderer.renderToken(tokens,index,options);
  const originalLink=md.renderer.rules.link_open||((tokens,index,options,env,renderer)=>renderer.renderToken(tokens,index,options));
  md.renderer.rules.link_open=(tokens,index,options,env,renderer)=>{const token=tokens[index],href=token.attrGet('href');if(authoring)token.attrSet('data-source-href',href);let target=null,anchor='';
    const sourcePrefix=content.product.repository+'/blob/main/';
    const sourceHref=href?.startsWith(sourcePrefix)?href.slice(sourcePrefix.length):href;
    if(sourceHref&&!/^(https?:|mailto:|#)/.test(sourceHref)){const split=sourceHref.split('#');anchor=split[1]||'';const base=href.startsWith(sourcePrefix)?'https://source.invalid/':`https://source.invalid/${page.source}`;const resolved=new URL(split[0],base).pathname.slice(1);target=pages.find(p=>p.source===resolved);}
    if(target){token.attrSet('href',target.url+(anchor?`#${anchor}`:''));token.attrSet('id',`link-${hash(page.source+':'+href+':'+linkCount++).slice(0,12)}`);if(target.part==='Reference')token.attrSet('data-reference','');}
    else if(href&&!/^(https?:|mailto:|#|\/)/.test(href))token.attrSet('href',`${content.product.repository}/blob/${evidence.revision.head}/${new URL(href,`https://source.invalid/${page.source}`).pathname.slice(1)}`);
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
    const observed=before?.type==='html_block'?before.content.match(/<!-- example-output: (\S+) source=(\S+) scenario=(\S+) interface=(\S+) role=observed -->/):null;
    const reference=before?.type==='html_block'?before.content.match(/<!-- reference: (\S+) source=(\S+) region=(\S+) role=(signature|definition|layout) -->/):null;
    const display=observed?evidence.displays.find(item=>item.id===observed[1]):null;
    if(!authoring&&observed&&(!display||display.sha256!==hash(source)))throw Error('Changed retained output: '+observed[1]);
    let record=null;if(marker){record=evidence.sources.find(s=>s.path===marker[2]&&s.region===marker[3]);if(!authoring&&(!record||record.sha256!==hash(source)))throw Error(`Missing or changed executable binding: ${marker[1]}`);}
    if(reference){record=evidence.sources.find(item=>item.path===reference[2]&&item.region===reference[3]);if(!authoring&&(!record||record.sha256!==hash(source)))throw Error('Missing or changed package reference: '+reference[1]);}
    const file=evidence.sources.find(item=>item.path===(display?.path||page.source)&&item.region==='__file__');
    const sourcePath=record?.path||display?.path||page.source,commit=record?.commit||file?.commit||evidence.revision.head;
    const origin=record||file;
    const identity=(display?`Observed in <a href="${content.product.repository}/commit/${display.retained_commit}">${display.retained_commit.slice(0,8)}</a> · `:'')+(origin?.status==='merged'?`<a href="${escape(origin.merged_pr)}">Merged source PR</a>`:'Candidate source · unmerged');
    const validated=marker||observed?evidence.validation.scenarios.find(s=>s.id===(marker?marker[4]:observed[3])&&s.interface===(marker?marker[5]:observed[4])&&s.status==='passed'):null;
    if(!authoring&&marker&&!validated)throw Error(`No successful execution for ${marker[1]}`);
    const referenceClaim=reference?`<span class="claim">Package ${escape(reference[4])} · ${escape(content.product.name)} ${escape(evidence.package.version)} · <a href="${content.product.basePath}/evidence/current.json">current package validation</a> · <a href="${content.product.basePath}/source/${escape(sourcePath)}">package source</a></span>`:'<span class="claim">Reference excerpt · execution is not asserted by this block.</span>';
    if(authoring)return `<figure class="code" id="${id}"${marker?` data-research-interface="${escape(marker[5])}"`:observed?` data-research-interface="${escape(observed[4])}"`:''} data-source="${Buffer.from(source).toString('base64')}"><figcaption class="code-bar"><span>${escape(language)}</span><span><button type="button" data-wrap aria-pressed="true" hidden>Wrap</button> <button type="button" data-copy hidden>Copy</button></span></figcaption><pre><code>${lines}</code></pre><div class="source">${record?.sha256===hash(source)||display?.sha256===hash(source)?`Historical excerpt verification: ${escape(evidence.package.version)} · ${escape(evidence.revision.head.slice(0,8))}`:'Not verified at this release'} · authoring branch execution is not asserted.</div></figure>`;
    return `<figure class="code" id="${id}"${marker?` data-research-interface="${escape(marker[5])}"`:observed?` data-research-interface="${escape(observed[4])}"`:""} data-source="${Buffer.from(source).toString('base64')}"><figcaption class="code-bar"><span>${escape(language)}${record?' · '+escape(record.path.split('/').at(-1)):''}</span><span><button type="button" data-wrap aria-pressed="true" hidden>Wrap</button> <button type="button" data-copy hidden>Copy</button></span></figcaption><pre><code>${lines}</code></pre><div class="source">${identity} · <a href="${content.product.repository}/blob/${commit}/${escape(sourcePath)}">${escape(commit.slice(0,8))}</a>${validated?`<span class="claim">${display?'Current validation':'Executed'} with ${escape(content.product.name)} ${escape(evidence.package.version)} · <a href="${content.product.basePath}/evidence/current.json">validation, inputs and tests</a> · <a href="${content.product.basePath}/source/${escape(sourcePath)}">${display?'retained output':'complete source'}</a> · <a href="${content.product.basePath}/evidence/${escape(evidence.bundle.path.split('/').at(-1))}">project and recorded inputs</a> · <a href="${content.product.basePath}/evidence/${escape(evidence.native_outputs.path.split('/').at(-1))}">native research records</a></span>`:referenceClaim}</div></figure>`;
  };
  const originalTableOpen=md.renderer.rules.table_open||(()=>'<table>');const originalTableClose=md.renderer.rules.table_close||(()=>'</table>');
  md.renderer.rules.table_open=(...args)=>'<div class="table-scroll" tabindex="0" role="region" aria-label="Scrollable reference table">'+originalTableOpen(...args);
  md.renderer.rules.table_close=(...args)=>originalTableClose(...args)+'</div>';
  const {body,kept,introduction,introductionSource}=documentText(page.markdown,page);if(authoring)authoring.annotate(md,kept);const introductionHtml=md.renderInline(introductionSource);return {html:md.render(body),headings,introduction,introductionHtml};
}
export function renderPage(page,pages,evidence,{product=content.product.id,results={},authoring}={}){
  const {html,headings,introduction,introductionHtml}=renderMarkdown(page,pages,evidence,results,authoring);const book=pages.filter(p=>p.bookOrder!==undefined).sort((a,b)=>a.bookOrder-b.bookOrder);const at=book.indexOf(page),next=at>=0?book[at+1]:null,previous=at>0?book[at-1]:null;
  const home=book[0];const reference=pages.find(p=>p.route==='/reference/'),developer=pages.find(p=>p.source==='docs/Developer/README.md');
  const navigation=page.part==='Book'?book:pages.filter(p=>p.part===page.part&&p.source).sort((a,b)=>pageLabel(a).localeCompare(pageLabel(b)));
  const item=p=>`<li><a href="${p.url}"${p===page?' aria-current="page"':''}>${escape(pageLabel(p))}</a></li>`;
  const groups=new Map();for(const entry of navigation){const group=entry.navigationGroup||page.part;if(!groups.has(group))groups.set(group,[]);groups.get(group).push(entry);}
  const nav=page.part==='Book'?navigation.map(item).join('')+pages.filter(p=>p.part==='Book'&&p.bookOrder===undefined&&p.source).map(item).join(''):[...groups].map(([group,entries])=>`<li class="nav-group"><details${entries.includes(page)?' open':''}><summary>${escape(group)}</summary><ul>${entries.map(item).join('')}</ul></details></li>`).join('');
  const contents=headings.map(h=>`<li><a href="#${escape(h.id)}">${escape(h.title)}</a></li>`).join('');
  const mobile=`<details class="mobile-nav"><summary>Contents</summary><nav aria-label="Chapter navigation"><p class="label">${page.part==='Book'?'The research book':escape(page.part)}</p><ul>${nav}</ul></nav><nav aria-label="In this chapter"><p class="label">In this chapter</p><ol>${contents}</ol></nav></details>`;
  const firstParagraph=introduction||page.title;
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${escape(page.title)} — ${escape(content.product.name)}</title><meta name="description" content="${escape(firstParagraph)}"><link rel="canonical" href="${content.product.origin}${page.url}"><link rel="stylesheet" href="${content.product.basePath}/assets/book.css"><script type="module" src="${content.product.basePath}/assets/main.mjs"></script></head><body data-part="${escape(page.part)}" data-product="${product}" data-content="${escape(page.source||'index:'+page.part)}" data-interface="both" data-search="${content.product.basePath}/search.json"><a class="skip" href="#main">Skip to the chapter</a><header class="site-header"><div class="header-inner"><a class="wordmark" href="${home.url}">${escape(content.product.wordmark)}</a><nav class="part-nav" aria-label="Parts"><a href="${home.url}"${page.part==='Book'?' aria-current="true"':''}>Book</a><a href="${reference.url}"${page.part==='Reference'?' aria-current="true"':''}>Reference</a><a href="${developer.url}"${page.part==='Developer'?' aria-current="true"':''}>Developer</a><button type="button" data-search-open hidden>Search</button></nav></div><div class="reading-progress"></div></header><div class="hero"><div class="hero-inner"><p class="label">${escape(page.part)} / ${escape(content.product.subject)}</p><h1 id="${escape([...headingIdentities(new MarkdownIt({html:true}).parse(page.markdown,{})).values()][0]?.id||slug(page.title))}" tabindex="-1">${escape(page.title)}</h1>${introduction?`<p class="hero-lead">${introductionHtml}</p>`:''}</div></div><div class="layout"><aside class="chapter-nav" aria-label="${page.part==='Book'?'Book chapters':escape(page.part)+' navigation'}"><p class="label">${page.part==='Book'?'The research book':escape(page.part)}</p><ul>${nav}</ul><nav class="chapter-contents" aria-label="In this chapter"><p class="label">In this chapter</p><ol>${contents}</ol></nav></aside><main id="main" tabindex="-1">${mobile}<nav class="interface-nav" aria-label="Research interfaces" data-interface-controls hidden><button type="button" data-interface-choice="both" aria-pressed="true">Both interfaces</button><button type="button" data-interface-choice="python" aria-pressed="false">Python Research</button><button type="button" data-interface-choice="compiled" aria-pressed="false">Compiled Research</button></nav><article>${html}</article><p class="status" data-status role="status" aria-live="polite"></p><div class="page-end"><span>${previous?`Previous<a href="${previous.url}">${escape(previous.title)}</a>`:escape(content.product.interfaces)}</span><span>${next?`Continue<a href="${next.url}">${escape(next.title)}</a>`:`<a href="${home.url}">Return to the research book</a>`}</span></div><div class="page-end">${page.source?`<a href="${content.product.repository}/edit/main/${escape(page.source)}">Edit this source</a>`:'Generated from the canonical source map'}<span>Source revision · ${escape(evidence.revision.head.slice(0,8))}</span></div></main></div><dialog class="reference-dialog" data-reference-dialog aria-label="Reference"><header class="dialog-head"><span class="label">Reference / keep your place</span><button type="button" data-reference-close>Return to chapter</button></header><div class="return-rule"></div><div class="reference-body" data-reference-body></div></dialog><dialog class="search-dialog" data-search-dialog aria-label="Search ${escape(content.product.name)}"><header class="dialog-head"><span class="label">Find a passage</span><button type="button" data-search-close>Close</button></header><div class="search-body"><label for="search">Search the book and references</label><input id="search" type="search" autocomplete="off"><p data-search-count class="status" aria-live="polite"></p><ol></ol></div></dialog></body></html>`;
}
