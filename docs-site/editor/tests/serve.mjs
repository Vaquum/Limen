import https from 'node:https';
import fs from 'node:fs/promises';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import worker,{EditorState} from '../worker.mjs';
import {protocol,Storage} from './protocol.mjs';
const site=path.resolve(import.meta.dirname,'../..'),temporary=path.resolve(site,'../.docs-evidence/editor-browser');await fs.mkdir(temporary,{recursive:true});
const cert=path.join(temporary,'certificate.pem'),key=path.join(temporary,'key.pem');
execFileSync('openssl',['req','-x509','-newkey','ec','-pkeyopt','ec_paramgen_curve:P-256','-nodes','-keyout',key,'-out',cert,'-days','1','-subj','/CN=localhost','-addext','subjectAltName=IP:127.0.0.1,DNS:localhost'],{stdio:'inherit'});
const nativeFetch=globalThis.fetch,publicRoot='https://docs.vaquum.fi/limen/',options={headers:{'User-Agent':'Limen-documentation-editor/1.0'},signal:AbortSignal.timeout(15000)};
const manifestReply=await nativeFetch(publicRoot+'publication.json',options);if(!manifestReply.ok)throw Error('Public publication unavailable: '+manifestReply.status);const manifestBytes=await manifestReply.text(),manifest=JSON.parse(manifestBytes),evidenceReply=await nativeFetch(publicRoot+manifest.artifacts.evidence.path,options);if(!evidenceReply.ok)throw Error('Public evidence unavailable: '+evidenceReply.status);const evidenceBytes=await evidenceReply.text();
let remote=await protocol(),objects=new Map(),origin,evidenceFailure=null;
globalThis.fetch=async function(url,init){
  if(this!==undefined&&this!==globalThis)throw Error('Worker fetch requires its global receiver');
  if(String(url).startsWith(publicRoot)&&init.redirect!=='manual')throw Error('Worker evidence fetch requires explicit manual redirect rejection');
  if(String(url)===publicRoot+'publication.json'){if(evidenceFailure)return new Response('Unavailable',{status:evidenceFailure==='redirect'?302:503,headers:{Location:'https://untrusted.invalid/evidence'}});return new Response(manifestBytes);}
  if(String(url)===publicRoot+manifest.artifacts.evidence.path)return new Response(evidenceBytes);
  if(String(url)==='https://github.com/login/oauth/access_token')return new Response(JSON.stringify({access_token:'browser-protocol-token'}));
  if(String(url).endsWith('/applications/browser-client/token'))return new Response(null,{status:204});
  return remote.transport(url,init);
};
const mime={'.html':'text/html; charset=utf-8','.mjs':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.woff2':'font/woff2'};
const env={GITHUB_CLIENT_ID:'browser-client',GITHUB_CLIENT_SECRET:'browser-client-secret',SESSION_KEY:Buffer.alloc(32,4).toString('base64'),EDITOR_STATE:{idFromName:name=>name,get(name){if(!objects.has(name))objects.set(name,new EditorState({storage:new Storage()}));return {fetch:(url,init)=>objects.get(name).fetch(new Request(url,init))};}},ASSETS:{async fetch(request){const base=path.join(site,'editor-build'),target=path.resolve(base,'.'+decodeURIComponent(new URL(request.url).pathname));if(!target.startsWith(base+path.sep))return new Response('Not found',{status:404});try{const file=(await fs.stat(target)).isDirectory()?path.join(target,'index.html'):target;return new Response(await fs.readFile(file),{headers:{'Content-Type':mime[path.extname(file)]||'application/octet-stream'}});}catch(error){if(error.code==='ENOENT')return new Response('Not found',{status:404});throw error;}}}};
const server=https.createServer({cert:await fs.readFile(cert),key:await fs.readFile(key)},async(incoming,outgoing)=>{
  try{
    const chunks=[];for await(const chunk of incoming)chunks.push(chunk);const body=Buffer.concat(chunks);
    if(incoming.url==='/__test/control'){
      const control=body.length?JSON.parse(body.toString()):{};
      if(control.action==='reset'){remote=await protocol();objects=new Map();evidenceFailure=null;}
      if(control.action==='evidence-failure')evidenceFailure=control.kind;
      if(control.action==='lose')remote.setLose(control.kind);if(control.action==='revoke')remote.revoke();
      const changed=[];for(const [branch,head] of remote.refs)if(branch!=='main'){const baseline=remote.commits.get(remote.refs.get('main')).files;for(const [source,text] of remote.commits.get(head).files)if(text!==baseline.get(source))changed.push(source);}
      outgoing.writeHead(200,{'Content-Type':'application/json'});outgoing.end(JSON.stringify({counts:remote.counts,pulls:remote.pulls,issues:remote.issues,changed}));return;
    }
    const request=new Request(origin+incoming.url,{method:incoming.method,headers:incoming.headers,...(['GET','HEAD'].includes(incoming.method)?{}:{body})}),reply=await worker.fetch(request,{...env,EDITOR_ORIGIN:origin});
    const headers=Object.fromEntries([...reply.headers].filter(([key])=>key!=='set-cookie'));outgoing.writeHead(reply.status,{...headers,...(reply.headers.getSetCookie().length?{'Set-Cookie':reply.headers.getSetCookie()}:{} )});outgoing.end(Buffer.from(await reply.arrayBuffer()));
  }catch(error){console.error(error);outgoing.writeHead(500);outgoing.end(error.message);}
});
server.listen(Number(process.env.EDITOR_TEST_PORT||4335),'127.0.0.1',()=>{origin='https://127.0.0.1:'+server.address().port;console.log('Authoring protocol fixture: '+origin);if(process.send)process.send({port:server.address().port});});
