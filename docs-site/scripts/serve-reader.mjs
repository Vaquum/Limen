import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const site=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const workerSource=await fs.readFile(path.join(site,'src/worker.js'),'utf8');
const {default:worker}=await import('data:text/javascript;base64,'+Buffer.from(workerSource).toString('base64'));
const output=path.join(site,'reader-build');const port=Number(process.env.READER_PORT||4330);
const mime={'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.mjs':'text/javascript; charset=utf-8','.json':'application/json; charset=utf-8','.xml':'application/xml','.woff2':'font/woff2','.py':'text/plain; charset=utf-8','.yaml':'text/plain; charset=utf-8'};
const assets={async fetch(request){const url=new URL(request.url),name=decodeURIComponent(url.pathname);if(process.env.READER_TEST_FIXTURES==='1'&&name==='/__test/other-product/'){const html=(await fs.readFile(path.join(output,'index.html'),'utf8')).replace('data-product="limen"','data-product="fixture-product"');return new Response(html,{headers:{'Content-Type':'text/html; charset=utf-8'}});}const target=path.resolve(output,'.'+name);if(!target.startsWith(output+path.sep)&&target!==output)return new Response('Not found',{status:404});let file=target;
  try{const info=await fs.stat(file);if(info.isDirectory())file=path.join(file,'index.html');const bytes=await fs.readFile(file);return new Response(bytes,{headers:{'Content-Type':mime[path.extname(file)]||'application/octet-stream'}});}
  catch(error){if(error.code==='ENOENT'||error.code==='ENOTDIR')return new Response('Not found',{status:404});throw error;}
}};
const server=http.createServer(async(incoming,outgoing)=>{try{const response=await worker.fetch(new Request(`http://127.0.0.1:${port}${incoming.url}`,{method:incoming.method}),{ASSETS:assets});outgoing.writeHead(response.status,Object.fromEntries(response.headers));outgoing.end(Buffer.from(await response.arrayBuffer()));}catch(error){console.error(error);outgoing.writeHead(500);outgoing.end('Reader server failed');}});
server.listen(port,'127.0.0.1',()=>console.log(`Candidate reader: http://127.0.0.1:${port}/limen/`));
