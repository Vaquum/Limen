import fs from 'node:fs/promises';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
import {content,pageLabel,renderPage} from '../reader/render.mjs';
import {documentTitle} from '../reader/headings.mjs';
import {compatibilityMarkdown,compatibilityContents} from './render-compat-input.mjs';
const root=path.resolve('..'),temporary=path.resolve('.docs-evidence/renderer-compat');
await fs.mkdir(temporary,{recursive:true});
await fs.writeFile(path.join(temporary,'render.mjs'),execFileSync('git',['show','616d270e6f69ca723456104c66207c55ee808021:docs-site/reader/render.mjs'],{cwd:root}));
await fs.writeFile(path.join(temporary,'content.json'),JSON.stringify(content));
const previous=await import(pathToFileURL(path.join(temporary,'render.mjs'))),evidence=JSON.parse(await fs.readFile('../.docs-evidence/current.json','utf8')),results={},catalogue=JSON.parse(await fs.readFile('../docs/examples/catalogue.json','utf8'));
for(const scenario of evidence.validation.scenarios)results[scenario.interface+'/'+scenario.id]={value:JSON.parse(await fs.readFile(path.join(root,scenario.output_path),'utf8')),source:catalogue.scenario_details[scenario.interface+'/'+scenario.id].source};
const pages=await Promise.all(Object.entries(content.pages).map(async([source,settings])=>{const markdown=compatibilityMarkdown(await fs.readFile(path.join(root,source),'utf8'));return {source,...settings,markdown,url:content.product.basePath+settings.route,title:documentTitle(markdown)||content.product.name};}));
const groups=new Map();for(const page of pages.filter(page=>page.part==='Reference').sort((a,b)=>pageLabel(a).localeCompare(pageLabel(b)))){const group=page.navigationGroup||'Lookup';if(!groups.has(group))groups.set(group,[]);groups.get(group).push(page);}
const links=[...groups].map(([group,entries])=>'## '+group+'\n\n'+entries.map(page=>`- [${pageLabel(page)}](${page.url})`).join('\n')).join('\n\n');
pages.push({source:null,route:'/reference/',url:content.product.basePath+'/reference/',part:'Reference',title:'Reference',markdown:'# Reference\n\nNames, parameters, return values and the boundaries of the two research interfaces.\n\n'+links+'\n'});
const changed=[];for(const page of pages){if(compatibilityContents(previous.renderPage(page,pages,evidence,{results}),page.markdown)!==renderPage(page,pages,evidence,{results}))changed.push(page.source||page.route);}
if(changed.length)throw Error('Public rendering changed for untouched authoring inputs: '+changed.join(', '));
console.log('Public renderer compatibility holds across '+pages.length+' current-source pages; retained heading identities are verified separately.');
