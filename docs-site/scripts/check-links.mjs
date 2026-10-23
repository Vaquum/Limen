import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import MarkdownIt from 'markdown-it';
import {content} from '../reader/render.mjs';

import {headingIdentities} from '../reader/headings.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const md=new MarkdownIt({html:true}),pages=new Map(),broken=[];
for(const [source,settings] of Object.entries(content.pages)){
  const tokens=md.parse(await fs.readFile(path.join(root,source),'utf8'),{});
  const anchors=new Set(),identities=headingIdentities(tokens),links=[];let code=0;
  for(let index=0;index<tokens.length;index++){
    const token=tokens[index];
    if(token.type==='heading_open'){
      anchors.add(identities.get(index).id);
    }
    if(token.type==='fence'){
      code++;anchors.add('code-'+code);
      for(let line=1;line<=token.content.trimEnd().split('\n').length;line++)anchors.add('code-'+code+'-L'+line);
    }
    if(token.type==='html_block'){
      for(const match of token.content.matchAll(/id="([^"]+)"|<!-- result: (\S+)/g))anchors.add(match[1]||match[2]);
    }
    for(const child of token.children||[])if(child.type==='link_open')links.push(child.attrGet('href'));
  }
  pages.set(source,{settings,anchors,links});
}
const routes=new Map([...pages].map(([source,page])=>[content.product.basePath+page.settings.route,source]));
for(const [source,{links}] of pages){
  for(let href of links){
    const prefix=content.product.repository+'/blob/main/',repository=href.startsWith(prefix);
    if(repository)href=href.slice(prefix.length);
    const target=new URL(href,repository?'https://source.invalid/':'https://source.invalid/'+source);
    const local=target.origin==='https://source.invalid';
    const hosted=target.origin===content.product.origin;
    const destination=local&&pages.has(target.pathname.slice(1))?target.pathname.slice(1):local||hosted?routes.get(target.pathname):null;
    if(!destination||!target.hash)continue;
    const anchor=decodeURIComponent(target.hash.slice(1)).replace(/-L(\d+)-L\d+$/,'-L$1');
    if(!pages.get(destination).anchors.has(anchor))broken.push(source+' → '+destination+target.hash);
  }
}
if(broken.length)throw Error('Broken authored source fragments:\n'+broken.join('\n'));
console.log(`Authored fragments resolve across ${pages.size} reader sources.`);
