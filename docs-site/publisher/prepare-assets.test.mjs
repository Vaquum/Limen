import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,writeFile,mkdir,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {prepareAssets,MAX_ASSET_BYTES} from './prepare-assets.mjs';

const bytes=await readFile(new URL('../../docs/examples/python-research/research.py',import.meta.url));
const sha256=createHash('sha256').update(bytes).digest('hex');
const relative=`evidence/rollback-${sha256}.tar.gz`;
async function fixture(t){
  const directory=await mkdtemp(path.join(tmpdir(),'limen-publication-assets-'));
  t.after(()=>rm(directory,{recursive:true}));
  await mkdir(path.join(directory,'evidence'));
  await writeFile(path.join(directory,relative),bytes);
  await writeFile(path.join(directory,'publication.json'),JSON.stringify({schema_version:1,release:'v5.46.2',artifacts:{rollback:{path:relative,sha256}}}));
  assert.ok((await readFile(path.join(directory,'publication.json'))).length<bytes.length-1);
  return directory;
}
test('normal assets remain byte-identical and require no network',async t=>{
  const directory=await fixture(t);
  assert.equal(MAX_ASSET_BYTES,26214400);
  assert.deepEqual(await prepareAssets(directory,{fetchPublic:()=>{throw new Error('No network expected');}}),{redirected:[]});
  assert.deepEqual(await readFile(path.join(directory,relative)),bytes);
  await assert.rejects(readFile(path.join(directory,'_redirects')),/ENOENT/);
});
test('only the verified oversized rollback receives an immutable anonymous redirect',async t=>{
  const directory=await fixture(t),before=await readFile(path.join(directory,'publication.json'));
  const requests=[];
  const result=await prepareAssets(directory,{limit:bytes.length-1,fetchPublic:async(url,options)=>{requests.push({url,options});return new Response(bytes);}});
  const url=`https://github.com/Vaquum/Limen/releases/download/v5.46.2/rollback-${sha256}.tar.gz`;
  assert.equal(requests[0].url,url);
  assert.deepEqual(Object.keys(requests[0].options.headers),['User-Agent']);
  assert.equal(await readFile(path.join(directory,'_redirects'),'utf8'),`/${relative} ${url} 302\n`);
  assert.equal(await readFile(path.join(directory,'.assetsignore'),'utf8'),`/${relative}\n`);
  assert.deepEqual(await readFile(path.join(directory,relative)),bytes);
  assert.deepEqual(await readFile(path.join(directory,'publication.json')),before);
  assert.equal(result.redirected[0].sha256,sha256);
});
for(const failure of ['changed','unavailable','local-change','other-member','existing-metadata'])test(`refuse ${failure} without excluding sealed assets`,async t=>{
  const directory=await fixture(t);
  if(failure==='local-change')await writeFile(path.join(directory,relative),Buffer.concat([bytes,bytes]));
  if(failure==='other-member')await writeFile(path.join(directory,'oversized.html'),bytes);
  if(failure==='existing-metadata')await writeFile(path.join(directory,'_redirects'),'existing\n');
  const fetchPublic=async()=>failure==='unavailable'?new Response(null,{status:404}):new Response(failure==='changed'?Buffer.concat([bytes,bytes]):bytes);
  const expected={changed:/Anonymous Release rollback differs/,unavailable:/Anonymous Release rollback differs/,'local-change':/Sealed rollback bytes changed/,'other-member':/oversized reader member/,'existing-metadata':/Provider metadata already exists/};
  await assert.rejects(prepareAssets(directory,{limit:bytes.length-1,fetchPublic}),expected[failure]);
  await assert.rejects(readFile(path.join(directory,'.assetsignore')),/ENOENT/);
});
