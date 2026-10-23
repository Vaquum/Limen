import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {parseArgs} from 'node:util';
import {execFileSync} from 'node:child_process';
import {content,renderPage,hash,pageLabel} from '../reader/render.mjs';
import {documentTitle} from '../reader/headings.mjs';
import {searchEntries} from '../reader/search-index.mjs';
const site=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),root=path.dirname(site);
const {values}=parseArgs({options:{evidence:{type:'string',default:'../.docs-evidence/current.json'},output:{type:'string',default:'reader-build'}}});
const dirty=execFileSync('git',['status','--porcelain','--untracked-files=normal'],{cwd:root,encoding:'utf8'});
if(dirty.trim())throw Error('Reader proof requires a clean published source tree');
const evidencePath=path.resolve(site,values.evidence);const evidence=JSON.parse(await fs.readFile(evidencePath,'utf8'));
const expectedHead=process.env.LIMEN_EXPECTED_HEAD||execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim();
const expectedTree=process.env.LIMEN_EXPECTED_TREE||execFileSync('git',['rev-parse','HEAD^{tree}'],{cwd:root,encoding:'utf8'}).trim();
if(evidence.revision.head!==expectedHead||evidence.revision.tree!==expectedTree||evidence.validation.status!=='passed')throw Error('Reader evidence is absent, failed or belongs to another source tree');
const packageText=await fs.readFile(path.join(root,'pyproject.toml'),'utf8');const version=packageText.match(/^version = "([^"]+)"/m)[1];
if(evidence.package.version!==version)throw Error('Reader evidence belongs to a different package version');
execFileSync(process.env.LIMEN_PYTHON||'python3',[path.join(root,'docs/examples/check_evidence.py'),'--evidence',evidencePath,'--head',expectedHead,'--tree',expectedTree,'--version',version],{cwd:root,stdio:'inherit'});
const overrides=content.pages;
const routes=Object.values(overrides).map(page=>page.route);
if(new Set(routes).size!==routes.length||routes.some(route=>!route.startsWith('/')||!route.endsWith('/')||route.includes('..')))throw Error('Canonical reader routes must be unique absolute directory paths');
if(Object.values(overrides).some(page=>page.navigationTitle!==undefined&&(typeof page.navigationTitle!=='string'||!page.navigationTitle.trim()||page.navigationTitle!==page.navigationTitle.trim())))throw Error('Navigation titles must be nonempty plain labels');
const pages=await Promise.all(Object.entries(overrides).map(async([source,chosen])=>{const markdown=await fs.readFile(path.join(root,source),'utf8');return {source,...chosen,markdown,url:content.product.basePath+chosen.route,title:documentTitle(markdown)||content.product.name};}));
const referenceGroups=new Map();for(const page of pages.filter(p=>p.part==='Reference').sort((a,b)=>pageLabel(a).localeCompare(pageLabel(b)))){const group=page.navigationGroup||'Lookup';if(!referenceGroups.has(group))referenceGroups.set(group,[]);referenceGroups.get(group).push(page);}
const referenceLinks=[...referenceGroups].map(([group,entries])=>'## '+group+'\n\n'+entries.map(p=>`- [${pageLabel(p)}](${p.url})`).join('\n')).join('\n\n');
pages.push({source:null,route:'/reference/',url:content.product.basePath+'/reference/',part:'Reference',title:'Reference',markdown:'# Reference\n\nNames, parameters, return values and the boundaries of the two research interfaces.\n\n'+referenceLinks+'\n'});
const output=path.resolve(site,values.output);if(output===site||!output.startsWith(site+path.sep))throw Error('Reader output must be inside the site directory');
const catalogue=JSON.parse(await fs.readFile(path.join(root,'docs/examples/catalogue.json'),'utf8')),results={};
for(const scenario of evidence.validation.scenarios){const identity=scenario.interface+'/'+scenario.id;const bytes=await fs.readFile(path.join(root,scenario.output_path));if(hash(bytes)!==scenario.output_sha256)throw Error('Changed scenario result: '+identity);results[identity]={value:JSON.parse(bytes),source:catalogue.scenario_details[identity].source};}
await fs.rm(output,{recursive:true,force:true});await fs.mkdir(path.join(output,'assets/fonts'),{recursive:true});
for(const page of pages){const target=path.join(output,page.route,'index.html');await fs.mkdir(path.dirname(target),{recursive:true});await fs.writeFile(target,renderPage(page,pages,evidence,{results}));}
await fs.writeFile(path.join(output,'404.html'),renderPage({source:null,route:'/404/',url:content.product.basePath+'/404/',part:'Book',title:'Page unavailable',markdown:'# Page unavailable\n\nThis address does not identify a page in the current book. [Return to the book]('+content.product.basePath+'/).\n'},pages,evidence));
for(const name of ['headings','main','motion','code','place','navigation','search','interfaces'])await fs.copyFile(path.join(site,'reader',name+'.mjs'),path.join(output,'assets',name+'.mjs'));
await fs.copyFile(path.join(site,'reader/book.css'),path.join(output,'assets/book.css'));
await fs.copyFile(path.join(site,'src/security-headers.json'),path.join(output,'security-headers.json'));
await fs.copyFile(path.join(site,'reader/route-decisions.json'),path.join(output,'route-decisions.json'));
for(const [from,to]of [['@fontsource-variable/archivo/files/archivo-latin-standard-normal.woff2','archivo.woff2'],['@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-400-normal.woff2','plex.woff2'],['@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-500-normal.woff2','plex-medium.woff2'],['@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-400-italic.woff2','plex-italic.woff2'],['@fontsource-variable/archivo/LICENSE','archivo-OFL.txt'],['@fontsource/ibm-plex-mono/LICENSE','plex-OFL.txt']])await fs.copyFile(path.join(site,'node_modules',from),path.join(output,'assets/fonts',to));
await fs.mkdir(path.join(output,'evidence'),{recursive:true});await fs.copyFile(evidencePath,path.join(output,'evidence/current.json'));
const bundleTarget=path.join(output,'evidence',path.basename(evidence.bundle.path));await fs.copyFile(path.resolve(root,evidence.bundle.path),bundleTarget);
await fs.copyFile(path.resolve(root,evidence.native_outputs.path),path.join(output,'evidence',path.basename(evidence.native_outputs.path)));
await fs.copyFile(path.join(root,'.docs-evidence/environment-wheels.json'),path.join(output,'evidence/environment-wheels.json'));
const wheelNames=(await fs.readdir(path.join(root,'.docs-evidence'))).filter(name=>name.startsWith('vaquum_limen-'+version+'-')&&name.endsWith('.whl'));
if(wheelNames.length!==1)throw Error('Expected exactly the validated wheel');
await fs.copyFile(path.join(root,'.docs-evidence',wheelNames[0]),path.join(output,'evidence',wheelNames[0]));
for(const source of [...evidence.sources,...evidence.displays]){const target=path.join(output,'source',source.path);await fs.mkdir(path.dirname(target),{recursive:true});await fs.copyFile(path.join(root,source.path),target);}
await fs.writeFile(path.join(output,'search.json'),JSON.stringify(pages.flatMap(searchEntries)));
await fs.writeFile(path.join(output,'routes.json'),JSON.stringify(pages.map(p=>({source:p.source,url:p.url,part:p.part}))));
await fs.writeFile(path.join(output,'sitemap.xml'),'<?xml version="1.0" encoding="utf-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'+pages.map(p=>`<url><loc>${content.product.origin}${p.url}</loc></url>`).join('')+'</urlset>');
await fs.writeFile(path.join(output,'robots.txt'),'User-agent: *\nAllow: '+content.product.basePath+'/\nSitemap: '+content.product.origin+content.product.basePath+'/sitemap.xml\n');
const members={};
async function collect(folder){for(const entry of await fs.readdir(folder,{withFileTypes:true})){const absolute=path.join(folder,entry.name);if(entry.isDirectory())await collect(absolute);else members[path.relative(output,absolute)]=hash(await fs.readFile(absolute));}}
await collect(output);
await fs.writeFile(path.join(output,'build.json'),JSON.stringify({head:expectedHead,tested_commit:evidence.revision.tested_commit,tree:expectedTree,version,evidence_sha256:hash(await fs.readFile(evidencePath)),files:members}));
console.log(`Reader assembled ${pages.filter(p=>p.source).length} maintained sources and ${pages.filter(p=>!p.source).length} generated indexes for ${expectedHead.slice(0,8)} / ${content.product.name} ${version}.`);
