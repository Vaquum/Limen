import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import MarkdownIt from 'markdown-it';
import CONTENT from '../../reader/content.json' with {type:'json'};
import {passages,applyPatches} from '../passages.mjs';
import {headingIdentities} from '../../reader/headings.mjs';
import {preview} from '../preview.mjs';
import {seal,unseal,challenge,jsonBody} from '../security.mjs';
import worker from '../worker.mjs';
const root=path.resolve(import.meta.dirname,'../../..'),blob='a'.repeat(40);
const patch=(passage,newMarkdown=passage.markdown)=>({...passage,newMarkdown,draftId:crypto.randomUUID()});
test('every mapped source round-trips every available prose passage without changing a byte',async()=>{
  let count=0;
  for(const source of Object.keys(CONTENT.pages)){
    const text=await fs.readFile(path.join(root,source),'utf8'),mapped=passages(text,source,blob);assert.ok(mapped.length,source);count+=mapped.length;
    for(let at=0;at<mapped.length;at+=512)assert.equal(applyPatches(text,source,blob,mapped.slice(at,at+512).map(item=>patch(item))),text,source);
  }
  assert.ok(count>1000);console.log('Exact no-op corpus passages:',count);
});
test('two length-changing edits in one actual page preserve every other source byte',async()=>{
  const source='docs/Developer/README.md',text=await fs.readFile(path.join(root,source),'utf8'),mapped=passages(text,source,blob),chosen=mapped.filter(item=>item.type==='paragraph').slice(0,2);
  assert.equal(chosen.length,2);let expected=text;for(const item of [...chosen].reverse())expected=expected.slice(0,item.sourceRange.start)+'Revised '+item.markdown+expected.slice(item.sourceRange.end);
  assert.equal(applyPatches(text,source,blob,chosen.map(item=>patch(item,'Revised '+item.markdown))),expected);
});
test('repeated prose has distinct source identities',async()=>{
  const source='docs/Built-In-SFDs.md',text=await fs.readFile(path.join(root,source),'utf8'),items=passages(text,source,blob).filter(item=>item.markdown==='It combines:');assert.equal(items.length,3);assert.equal(new Set(items.map(item=>item.id)).size,3);
  const result=applyPatches(text,source,blob,[patch(items[1],'It composes:')]);assert.equal(result,text.slice(0,items[1].sourceRange.start)+'It composes:'+text.slice(items[1].sourceRange.end));
});
test('heading renames preserve old duplicate and interface destinations',async()=>{
  const source='docs/Research-Book.md',text=await fs.readFile(path.join(root,source),'utf8'),mapped=passages(text,source,blob),chosen=mapped.find(item=>item.type==='heading'&&item.markdown.includes('Compiled'));assert.ok(chosen);
  const result=applyPatches(text,source,blob,[patch(chosen,'The compiled search declaration')]),md=new MarkdownIt({html:true});
  assert.deepEqual([...headingIdentities(md.parse(result,{})).values()].map(item=>item.id),[...headingIdentities(md.parse(text,{})).values()].map(item=>item.id));assert.match(result,/data-reader-heading="compiled"/);
});
test('stale blobs, modified ranges, overlap and structural/HTML attacks fail on the server',async()=>{
  const source='docs/Developer/README.md',text=await fs.readFile(path.join(root,source),'utf8'),item=passages(text,source,blob).find(item=>item.type==='paragraph');
  assert.throws(()=>applyPatches(text,source,'b'.repeat(40),[patch(item)]),/blob moved/);
  assert.throws(()=>applyPatches(text,source,blob,[{...patch(item),originalText:'wrong source'}]),/no longer matches/);
  assert.throws(()=>applyPatches(text,source,blob,[patch(item),patch(item)]),/Overlapping/);
  for(const value of ['# New section','- New item','one\n\ntwo','<img src=x onerror=alert(1)>','[new](javascript:alert(1))','[new](https://example.com/)'])assert.throws(()=>applyPatches(text,source,blob,[patch(item,value)]),undefined,value);
});
test('encrypted sessions are authenticated and key rotation invalidates them',async()=>{
  const key=Buffer.alloc(32,7).toString('base64'),value=await seal('private user token',key);assert.ok(!value.includes('private'));assert.equal(await unseal(value,key),'private user token');await assert.rejects(()=>unseal(value,Buffer.alloc(32,8).toString('base64')),/key was rotated/);assert.equal((await challenge('state verifier')).length,43);
});
test('direct requests enforce the authoring origin before asset routing',async()=>{
  let called=false;const env={EDITOR_ORIGIN:'https://limen-editor.vaquum.fi',ASSETS:{fetch(){called=true;return new Response('asset');}}};
  const reply=await worker.fetch(new Request('https://unexpected.workers.dev/limen/edit/'),env);assert.equal(reply.status,403);assert.equal(called,false);
  const unauth=await worker.fetch(new Request(env.EDITOR_ORIGIN+'/limen/edit/view/'),env);assert.equal(unauth.status,401);assert.match(unauth.headers.get('Content-Security-Policy'),/script-src 'self';/);assert.equal(unauth.headers.get('Cache-Control'),'no-store');
});
test('write origin, media type and streaming request limit are checked directly',async()=>{
  const origin='https://limen-editor.vaquum.fi';
  await assert.rejects(()=>jsonBody(new Request(origin+'/limen/edit/api/commit',{method:'POST',headers:{Origin:'https://docs.vaquum.fi','Content-Type':'application/json'},body:'{}'}),origin),/Exact authoring Origin/);
  await assert.rejects(()=>jsonBody(new Request(origin+'/limen/edit/api/commit',{method:'POST',headers:{Origin:origin,'Content-Type':'text/plain'},body:'{}'}),origin),/JSON is required/);
  await assert.rejects(()=>jsonBody(new Request(origin+'/limen/edit/api/commit',{method:'POST',headers:{Origin:origin,'Content-Type':'application/json'},body:' '.repeat(2097153)}),origin),/2 MiB/);
});
test('externally supplied HTML is inert in an authenticated authoring render',async()=>{
  const source='docs/Developer/README.md',markdown=(await fs.readFile(path.join(root,source),'utf8'))+'\n<img src=x onerror=alert(1)>\n<script>alert(1)</script>\n',page={source,markdown,title:'Developer',part:'Developer',route:'/developer/',url:'/limen/developer/'};
  const pages=[page,{source:'docs/Research-Book.md',title:'Book',part:'Book',bookOrder:0,url:'/limen/'},{source:null,route:'/reference/',title:'Reference',url:'/limen/reference/'}],evidence={package:{version:'5.46.2'},revision:{head:blob},sources:[],displays:[],validation:{scenarios:[]}};
  const html=preview(page,pages,evidence,{passages:passages(markdown,source,blob),state:{id:'workspace',branch:'prose',head:blob},pr:null});assert.ok(!html.includes('<img src=x'));assert.ok(!html.includes('<script>alert'));assert.match(html,/&lt;img src=x/);assert.match(html,/branch has not been executed/);
});
test('edited wraps, quoted introductions, nested lists, cells, references and summaries preserve their source surroundings',()=>{
  const source='docs/Developer/README.md',text='# Title\n\n> Quoted introduction\n> with **emphasis**.\n\n- Parent\n  - Nested `code` and [label][ref]\n\nHard wrapped\nparagraph with two spaces.  \nAnd a hard break.\n\n| Value | Meaning |\n| --- | --- |\n| `a\\|b` | &amp; and **bold** |\n\n[ref]: ../Research-Book.md "Book title"\n\n<details>\n<summary>Scope &amp; purpose</summary>\n\nDetail.\n\n</details>\n',mapped=passages(text,source,blob);
  for(const item of mapped){
    const value=item.type==='summary'?'Scope < extent & purpose':item.markdown.replace('Quoted','Revised').replace('Nested','Inner').replace('Hard wrapped','Soft wrapped').replace('Meaning','Definition').replace('**bold**','**clear**');
    const changed=applyPatches(text,source,blob,[patch(item,value)]);
    assert.ok(changed.startsWith(text.slice(0,item.sourceRange.start)));assert.ok(changed.endsWith(text.slice(item.sourceRange.end)));assert.equal(passages(changed,source,blob).length,mapped.length);
    if(item.type==='summary')assert.match(changed,/<summary>Scope &lt; extent &amp; purpose<\/summary>/);
  }
});
test('capitalization-only heading edits retain their research interface',async()=>{
  const source='docs/Research-Book.md',text=await fs.readFile(path.join(root,source),'utf8'),mapped=passages(text,source,blob),md=new MarkdownIt({html:true});
  for(const word of ['Compiled','Python']){const item=mapped.find(item=>item.type==='heading'&&item.markdown.includes(word)),changed=applyPatches(text,source,blob,[patch(item,item.markdown.replace(word,word.toLowerCase()))]);assert.deepEqual([...headingIdentities(md.parse(changed,{})).values()].map(item=>[item.id,item.kind]),[...headingIdentities(md.parse(text,{})).values()].map(item=>[item.id,item.kind]));}
});
test('canonical authoring links never replace their preserved source targets',async()=>{
  const source='docs/Developer/README.md',markdown=(await fs.readFile(path.join(root,source),'utf8'))+'\n[Book](/limen/)\n',page={source,markdown,title:'Developer',part:'Developer',route:'/developer/home/',url:'/limen/developer/home/'},book={source:'docs/Research-Book.md',title:'Book',part:'Book',bookOrder:0,route:'/',url:'/limen/'};
  const html=preview(page,[page,book,{source:null,route:'/reference/',title:'Reference',url:'/limen/reference/'}],{package:{version:'5.46.2'},revision:{head:blob},sources:[],displays:[],validation:{scenarios:[]}},{passages:passages(markdown,source,blob),state:{id:'delivery',head:blob,branch:'prose'},pr:41});assert.match(html,/data-source-href="\/limen\/"/);assert.ok(!html.includes('data-source-href="/limen/edit/view'));
});
