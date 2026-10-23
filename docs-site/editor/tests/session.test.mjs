import test from 'node:test';
import assert from 'node:assert/strict';
import worker,{EditorState} from '../worker.mjs';
import {protocol,Storage} from './protocol.mjs';
import {passages} from '../passages.mjs';
const origin='https://limen-editor.vaquum.fi',root=origin+'/limen/edit/';
test('actual handler binds single-use OAuth state/PKCE, rechecks permission and revokes copied session cookies',async()=>{
  const remote=await protocol(),objects=new Map(),originalFetch=globalThis.fetch;let verifier,revoked=false;
  const env={EDITOR_ORIGIN:origin,GITHUB_CLIENT_ID:'test-client',GITHUB_CLIENT_SECRET:'test-client-secret',SESSION_KEY:Buffer.alloc(32,9).toString('base64'),EDITOR_STATE:{idFromName:name=>name,get(name){if(!objects.has(name))objects.set(name,new EditorState({storage:new Storage()}));return {fetch:(url,init)=>objects.get(name).fetch(new Request(url,init))};}}};
  globalThis.fetch=async(url,options)=>{
    if(String(url)==='https://github.com/login/oauth/access_token'){const body=JSON.parse(options.body);verifier=body.code_verifier;return new Response(JSON.stringify({access_token:'technical-protocol-token'}));}
    if(String(url).endsWith('/applications/test-client/token')){assert.equal(options.headers['X-GitHub-Api-Version'],'2026-03-10');assert.equal(options.headers.Accept,'application/vnd.github+json');revoked=true;return new Response(null,{status:204});}
    return remote.transport(url,options);
  };
  try{
    const start=await worker.fetch(new Request(root+'auth/start?route=/developer/'),env);assert.equal(start.status,302);const authorization=new URL(start.headers.get('Location')),flowCookie=start.headers.get('Set-Cookie').split(';')[0],state=authorization.searchParams.get('state');
    assert.equal(authorization.searchParams.get('code_challenge_method'),'S256');assert.ok(state);
    const callback=await worker.fetch(new Request(root+'auth/callback?state='+state+'&code=protocol-code',{headers:{Cookie:flowCookie}}),env);assert.equal(callback.status,302);assert.ok(verifier);const digest=Buffer.from(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(verifier))).toString('base64url');assert.equal(authorization.searchParams.get('code_challenge'),digest);
    const copied=callback.headers.getSetCookie().find(value=>value.startsWith('__Host-limen-session=')).split(';')[0],request=path=>new Request(root+path,{headers:{Cookie:copied}});
    const signed=await worker.fetch(request('api/session'),env);assert.equal(signed.status,200);assert.equal((await worker.fetch(new Request(root+'auth/callback?state='+state+'&code=protocol-code',{headers:{Cookie:flowCookie}}),env)).status,403);
    const workspace=(await signed.json()).workspace,source='docs/Developer/README.md',page=(await remote.github.pages(workspace.head)).find(page=>page.source===source),first=passages(page.markdown,source,page.blob).find(item=>item.type==='paragraph');
    const post=(path,body)=>worker.fetch(new Request(root+'api/'+path,{method:'POST',headers:{Cookie:copied,Origin:origin,'Content-Type':'application/json'},body:JSON.stringify(body)}),env);
    const committed=await post('commit',{workspace:workspace.id,expectedHead:workspace.head,operation:crypto.randomUUID(),patches:[{...first,newMarkdown:first.markdown+' Reviewed.'}]});assert.equal(committed.status,200);
    const delivered=await committed.json(),current=(await remote.github.pages(delivered.head)).find(page=>page.source===source),original=passages(current.markdown,source,current.blob).find(item=>item.type==='paragraph'),files=new Map(remote.commits.get(delivered.head).files),advanced='b'.repeat(40);
    files.set(source,current.markdown.slice(0,original.sourceRange.start)+'Upstream revised this passage.'+current.markdown.slice(original.sourceRange.end)+'\n\n'+original.originalText+'\n');remote.commits.set(advanced,{files,parents:[delivered.head],message:'docs: revise the original passage'});remote.refs.set(delivered.branch,advanced);
    const refused=await post('reanchor',{patch:{...original,newMarkdown:'A local draft.'}});assert.equal(refused.status,409);assert.match((await refused.json()).error,/Source blob moved/);assert.equal(remote.counts.commit,1);
    remote.revoke();assert.equal((await worker.fetch(request('api/session'),env)).status,403);assert.equal((await worker.fetch(request('view/'),env)).status,403);
    const logout=await worker.fetch(new Request(root+'api/logout',{method:'POST',headers:{Cookie:copied,Origin:origin,'Content-Type':'application/json'},body:'{}'}),env);assert.equal(logout.status,200);assert.deepEqual(await logout.json(),{ok:true,revoked:true,revocationStatus:204});assert.equal(revoked,true);assert.equal((await worker.fetch(request('api/session'),env)).status,401);
  }finally{globalThis.fetch=originalFetch;}
});
