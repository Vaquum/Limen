import {createHash} from 'node:crypto';
import {readFile,writeFile,readdir,stat} from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

export const MAX_ASSET_BYTES=25*1024*1024;
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');

export async function prepareAssets(directory,{fetchPublic=fetch,limit=MAX_ASSET_BYTES}={}){
  const manifest=JSON.parse(await readFile(path.join(directory,'publication.json'),'utf8'));
  const rollback=manifest.artifacts.rollback;
  if(manifest.schema_version!==1||!/^v\d+\.\d+\.\d+$/.test(manifest.release)||!/^[a-f0-9]{64}$/.test(rollback.sha256)||rollback.path!==`evidence/rollback-${rollback.sha256}.tar.gz`)throw new Error('Invalid sealed rollback identity');
  const oversized=[];
  async function inspect(folder){
    for(const entry of await readdir(folder,{withFileTypes:true})){
      const file=path.join(folder,entry.name);
      if(entry.isSymbolicLink())throw new Error('Publication assets cannot contain symlinks');
      if(entry.isDirectory())await inspect(file);
      else if((await stat(file)).size>limit)oversized.push(path.relative(directory,file).split(path.sep).join('/'));
    }
  }
  await inspect(directory);
  if(!oversized.length)return {redirected:[]};
  if(oversized.length!==1||oversized[0]!==rollback.path)throw new Error('An oversized reader member requires a separate storage decision');
  for(const name of ['_redirects','.assetsignore']){
    try{await stat(path.join(directory,name));}catch(error){if(error.code==='ENOENT')continue;throw error;}
    throw new Error('Provider metadata already exists: '+name);
  }
  const bytes=await readFile(path.join(directory,rollback.path));
  if(digest(bytes)!==rollback.sha256)throw new Error('Sealed rollback bytes changed');
  const url=`https://github.com/Vaquum/Limen/releases/download/${manifest.release}/${path.posix.basename(rollback.path)}`;
  const response=await fetchPublic(url,{headers:{'User-Agent':'LimenDocsPublisher/1.0'},signal:AbortSignal.timeout(30000)});
  if(!response.ok||digest(Buffer.from(await response.arrayBuffer()))!==rollback.sha256)throw new Error('Anonymous Release rollback differs from the sealed bytes');
  await writeFile(path.join(directory,'_redirects'),`/${rollback.path} ${url} 302\n`,{flag:'wx'});
  await writeFile(path.join(directory,'.assetsignore'),`/${rollback.path}\n`,{flag:'wx'});
  return {redirected:[{path:rollback.path,url,sha256:rollback.sha256,bytes:bytes.length}]};
}

if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  if(process.argv.length!==3)throw new Error('Pass the sealed reader directory');
  console.log(JSON.stringify(await prepareAssets(path.resolve(process.argv[2])),null,2));
}
