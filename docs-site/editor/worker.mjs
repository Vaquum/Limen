import {GitHub,workspace,commit,openPull} from './github.mjs';
import {EditError,passages,applyPatches} from './passages.mjs';
import {random,challenge,seal,unseal,cookie,readCookie,headers,response,exactOrigin,jsonBody} from './security.mjs';
import {hash,escape} from '../reader/render.mjs';
import {preview,inlinePreview,referenceIndex,authoringSearch} from './preview.mjs';

const root='/limen/edit/';
const stateFor=(env,name)=>env.EDITOR_STATE.get(env.EDITOR_STATE.idFromName(name));
async function rpc(env,name,action,body={}){
  const result=await stateFor(env,name).fetch('https://state.invalid/'+action,{method:'POST',body:JSON.stringify(body)});
  const data=await result.json();if(!result.ok)throw new EditError(data.error,result.status);return data;
}
async function session(request,env,{permission=true}={}){
  const id=readCookie(request,'session');if(!id)throw new EditError('Sign in with GitHub',401);
  const record=await rpc(env,'auth','get',{id:'session:'+id});
  if(!record.value||record.value.expires<=Date.now())throw new EditError('Session expired; sign in again',401);
  const token=await unseal(record.value.token,env.SESSION_KEY),github=new GitHub(token);
  const user=permission?await github.writer():record.value.user;
  if(user.id!==record.value.user.id)throw new EditError('GitHub identity changed; sign in again',401);
  return {id,token,github,user};
}
const continuing=url=>{const pr=url.searchParams.get('pr');if(pr&&!/^[1-9]\d*$/.test(pr))throw new EditError('Invalid PR number',400);return pr?Number(pr):null;};
async function editing(request,env,url){
  const auth=await session(request,env),pr=continuing(url),name='writer:'+auth.user.id+(pr?':pr:'+pr:'');
  const state=await rpc(env,name,'workspace',{token:auth.token,user:auth.user,continuing:pr});return {...auth,state,name};
}
export class EditorState{
  constructor(state){this.storage=state.storage;this.queue=Promise.resolve();}
  async alarm(){let next;for(const [id,value] of await this.storage.list()){if(!value.expires)continue;if(value.expires<=Date.now())await this.storage.delete(id);else next=next===undefined?value.expires:Math.min(next,value.expires);}if(next!==undefined)await this.storage.setAlarm(next);}
  async fetch(request){
    const operation=this.queue.then(async()=>{
      try{
        const data=await request.json(),action=new URL(request.url).pathname.slice(1);
        if(action==='get')return response({value:await this.storage.get(data.id)||null});
        if(action==='put'){await this.storage.put(data.id,data.value);if(data.value.expires){const alarm=await this.storage.getAlarm();if(!alarm||data.value.expires<alarm)await this.storage.setAlarm(data.value.expires);}return response({ok:true});}
        if(action==='consume'){const value=await this.storage.get(data.id);await this.storage.delete(data.id);return response({value:value||null});}
        if(action==='delete'){await this.storage.delete(data.id);return response({ok:true});}
        const github=new GitHub(data.token),current=await workspace(this.storage,github,data.user,data.continuing);
        if(action==='workspace')return response(current);
        if(action==='commit')return response(await commit(this.storage,github,data.user,data.payload,current));
        if(action==='pr')return response(await openPull(this.storage,github,data.user,data.payload,current));
        throw new EditError('Unknown editing operation',404);
      }catch(error){return response({error:error.message},error instanceof EditError?error.status:502);}
    });
    this.queue=operation.then(()=>undefined,()=>undefined);return operation;
  }
}
export default{
  async fetch(request,env){
    try{
      exactOrigin(request,env.EDITOR_ORIGIN);const url=new URL(request.url),path=url.pathname;
      if(path===root+'auth/start'){
        if(!env.GITHUB_CLIENT_ID||!env.GITHUB_CLIENT_SECRET)throw new EditError('GitHub App installation is not configured',503);
        const id=random(),state=random(),verifier=random()+random(),pr=continuing(url),route=url.searchParams.get('route')||'/';
        await rpc(env,'auth','put',{id:'oauth:'+id,value:{state,verifier,pr,route,expires:Date.now()+600000}});
        const target=new URL('https://github.com/login/oauth/authorize');for(const [key,value] of Object.entries({client_id:env.GITHUB_CLIENT_ID,redirect_uri:env.EDITOR_ORIGIN+root+'auth/callback',state,code_challenge:await challenge(verifier),code_challenge_method:'S256'}))target.searchParams.set(key,value);
        return response('',302,{Location:target.href,'Set-Cookie':cookie('oauth',id,600)});
      }
      if(path===root+'auth/callback'){
        const flow=readCookie(request,'oauth'),record=flow?await rpc(env,'auth','consume',{id:'oauth:'+flow}):null;
        if(!record?.value||record.value.expires<Date.now()||record.value.state!==url.searchParams.get('state')||!url.searchParams.get('code'))throw new EditError('OAuth state is missing, expired or already used',403);
        const result=await fetch('https://github.com/login/oauth/access_token',{method:'POST',headers:{Accept:'application/json','Content-Type':'application/json'},body:JSON.stringify({client_id:env.GITHUB_CLIENT_ID,client_secret:env.GITHUB_CLIENT_SECRET,code:url.searchParams.get('code'),redirect_uri:env.EDITOR_ORIGIN+root+'auth/callback',code_verifier:record.value.verifier})});
        const token=await result.json();if(!result.ok||!token.access_token)throw new EditError('GitHub sign-in failed: '+(token.error_description||result.status),401);
        const user=await new GitHub(token.access_token).writer(),id=random();
        await rpc(env,'auth','put',{id:'session:'+id,value:{user,token:await seal(token.access_token,env.SESSION_KEY),expires:Date.now()+3600000}});
        const target=new URL(root,env.EDITOR_ORIGIN);target.searchParams.set('route',record.value.route);if(record.value.pr)target.searchParams.set('pr',record.value.pr);
        const reply=response('',302,{Location:target.href});reply.headers.append('Set-Cookie',cookie('session',id,3600));reply.headers.append('Set-Cookie',cookie('oauth','',0));return reply;
      }
      if(path===root+'api/logout'&&request.method==='POST'){
        await jsonBody(request,env.EDITOR_ORIGIN);const auth=await session(request,env,{permission:false});await rpc(env,'auth','delete',{id:'session:'+auth.id});
        let revoked;try{revoked=await fetch('https://api.github.com/applications/'+env.GITHUB_CLIENT_ID+'/token',{method:'DELETE',headers:{Authorization:'Basic '+Buffer.from(env.GITHUB_CLIENT_ID+':'+env.GITHUB_CLIENT_SECRET).toString('base64'),'Content-Type':'application/json','User-Agent':'Limen-docs-editor',Accept:'application/vnd.github+json','X-GitHub-Api-Version':'2026-03-10'},body:JSON.stringify({access_token:auth.token})});}catch(error){return response({ok:true,revoked:false,error:'Session invalidated; provider revocation failed'},200,{'Set-Cookie':cookie('session','',0)});}
        return response({ok:true,revoked:revoked.ok,revocationStatus:revoked.status},200,{'Set-Cookie':cookie('session','',0)});
      }
      if(path===root+'api/session'){
        const auth=await editing(request,env,url);return response({user:auth.user,workspace:auth.state,origin:env.EDITOR_ORIGIN});
      }
      if((path===root+'api/commit'||path===root+'api/pr')&&request.method==='POST'){
        const payload=await jsonBody(request,env.EDITOR_ORIGIN),auth=await editing(request,env,url);
        if(typeof payload.operation!=='string'||!/^[\da-f-]{36}$/.test(payload.operation))throw new EditError('An operation UUID is required');
        return response(await rpc(env,auth.name,path.endsWith('/commit')?'commit':'pr',{token:auth.token,user:auth.user,continuing:continuing(url),payload}));
      }
      if(path.startsWith(root+'view/')||path===root+'api/inline'||path===root+'api/reanchor'||path==='/limen/search.json'){
        const body=(path===root+'api/inline'||path===root+'api/reanchor')?await jsonBody(request,env.EDITOR_ORIGIN):null,auth=await editing(request,env,url),pages=await auth.github.pages(auth.state.head);
        if(body){const page=pages.find(page=>page.source===body.patch?.path);if(!page)throw new EditError('Unknown prose source',403);applyPatches(page.markdown,page.source,page.blob,[body.patch]);if(path.endsWith('/reanchor'))return response({patch:body.patch,head:auth.state.head});const target=passages(page.markdown,page.source,page.blob).find(item=>item.sourceRange.start===body.patch.sourceRange.start);return response({html:target.type==='summary'?escape(body.patch.newMarkdown):inlinePreview(body.patch.newMarkdown,page,pages,continuing(url))});}
        if(path==='/limen/search.json')return response(authoringSearch(pages,continuing(url)));
        pages.push(referenceIndex(pages));
        const route='/'+path.slice((root+'view/').length),page=pages.find(page=>page.route===route);
        if(!page)throw new EditError('This route is not an editable maintained page',404);
        const manifestResult=await fetch('https://docs.vaquum.fi/limen/publication.json',{headers:{'User-Agent':'Limen-documentation-editor/1.0'},redirect:'manual'});if(!manifestResult.ok)throw new EditError('Published evidence is unavailable',502);
        const publication=await manifestResult.json(),binding=publication.artifacts?.evidence;if(!binding||!/^evidence\/evidence-[a-f0-9]{64}\.json$/.test(binding.path))throw new EditError('Published evidence identity is invalid',502);
        const evidenceResult=await fetch('https://docs.vaquum.fi/limen/'+binding.path,{headers:{'User-Agent':'Limen-documentation-editor/1.0'},redirect:'manual'});if(!evidenceResult.ok)throw new EditError('Published evidence is unavailable',502);
        const evidenceBytes=Buffer.from(await evidenceResult.arrayBuffer());if(hash(evidenceBytes)!==binding.sha256)throw new EditError('Published evidence digest differs',502);const evidence=JSON.parse(evidenceBytes.toString());if(evidence.revision.head!==publication.merged_commit||'v'+evidence.package.version!==publication.release)throw new EditError('Published evidence revision differs',502);
        return response(preview(page,pages,evidence,{publication,state:auth.state,user:auth.user,pr:continuing(url),passages:page.source?passages(page.markdown,page.source,page.blob):[]}));
      }
      if(path===root||path.startsWith(root+'assets/')||path.startsWith('/limen/assets/')){
        const asset=await env.ASSETS.fetch(request);const reply=new Response(asset.body,asset);for(const [name,value] of Object.entries(headers))reply.headers.set(name,value);if(path===root)reply.headers.set('X-Frame-Options','DENY');return reply;
      }
      throw new EditError('Unknown editor route',404);
    }catch(error){return response({error:error.message},error instanceof EditError?error.status:502);}
  }
};
