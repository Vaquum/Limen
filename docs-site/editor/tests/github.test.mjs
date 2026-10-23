import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import CONTENT from '../../reader/content.json' with {type:'json'};
import {GitHub,workspace,commit,openPull,repository} from '../github.mjs';
import {passages} from '../passages.mjs';
const user={id:1,login:'writer'},hash=value=>createHash('sha1').update(value).digest('hex');
class Storage{constructor(){this.values=new Map();}async get(key){return structuredClone(this.values.get(key));}async put(key,value){this.values.set(key,structuredClone(value));}}
async function protocol(){
  const sources=new Map(await Promise.all(Object.keys(CONTENT.pages).map(async source=>[source,await fs.readFile(path.resolve(import.meta.dirname,'../../..',source),'utf8')]))),initial=hash('initial protocol revision');
  const commits=new Map([[initial,{files:sources,parents:[],message:'Initial protocol revision'}]]),refs=new Map([['main',initial]]),issues=[],pulls=[],counts={ref:0,commit:0,issue:0,pr:0};let lose=null,push=true;
  const json=(value,status=200)=>new Response(JSON.stringify(value),{status,headers:{'Content-Type':'application/json'}});
  async function transport(url,options={}){
    const route=new URL(url).pathname.slice(1),query=new URL(url).searchParams,body=options.body?JSON.parse(options.body):null;
    if(route==='user')return json(user);if(route==='repos/'+repository)return json({permissions:{push}});
    if(route.startsWith('repos/'+repository+'/git/ref/heads/')){const branch=decodeURIComponent(route.split('/heads/')[1]);return refs.has(branch)?json({object:{sha:refs.get(branch)}}):json({message:'Not found'},404);}
    if(route==='repos/'+repository+'/git/refs'){refs.set(body.ref.slice(11),body.sha);counts.ref++;if(lose==='ref'){lose=null;throw Error('Response lost after creating ref');}return json({object:{sha:body.sha}},201);}
    if(route.startsWith('repos/'+repository+'/git/commits/')){const value=commits.get(route.split('/').at(-1));return json({parents:value.parents.map(sha=>({sha})),message:value.message});}
    if(route==='graphql'){
      if(body.variables.input){const input=body.variables.input,branch=input.branch.branchName;if(refs.get(branch)!==input.expectedHeadOid)return json({errors:[{message:'Expected head mismatch'}]});const files=new Map(commits.get(input.expectedHeadOid).files);for(const addition of input.fileChanges.additions)files.set(addition.path,Buffer.from(addition.contents,'base64').toString());const message=input.message.headline+'\n\n'+input.message.body,oid=hash(input.expectedHeadOid+message+JSON.stringify([...files]));commits.set(oid,{files,parents:[input.expectedHeadOid],message});refs.set(branch,oid);counts.commit++;if(lose==='commit'){lose=null;throw Error('Response lost after commit');}return json({data:{createCommitOnBranch:{commit:{oid,url:'https://github.com/Vaquum/Limen/commit/'+oid}}}});}
      const result={};for(const [name,expression] of Object.entries(body.variables)){const at=expression.indexOf(':'),head=expression.slice(0,at),source=expression.slice(at+1),text=commits.get(head)?.files.get(source);result[name]=text===undefined?null:{oid:hash(text),text};}return json({data:{repository:result}});
    }
    if(route==='repos/'+repository+'/issues'){
      if(options.method==='POST'){const issue={...body,number:100+issues.length,state:'open'};issues.push(issue);counts.issue++;if(lose==='issue'){lose=null;throw Error('Response lost after issue');}return json(issue,201);}return json(issues);
    }
    if(route.startsWith('repos/'+repository+'/issues/'))return json(issues.find(issue=>issue.number===Number(route.split('/').at(-1))));
    if(route==='repos/'+repository+'/pulls'){
      if(options.method==='POST'){const pull={...body,number:200+pulls.length,state:'open',user,head:{ref:body.head,sha:refs.get(body.head),repo:{full_name:repository}},base:{ref:body.base,sha:refs.get('main'),repo:{full_name:repository}},html_url:'https://github.com/Vaquum/Limen/pull/'+(200+pulls.length)};pulls.push(pull);counts.pr++;if(lose==='pr'){lose=null;throw Error('Response lost after PR');}return json(pull,201);}return json(pulls.filter(pull=>'Vaquum:'+pull.head.ref===query.get('head')));
    }
    if(route.startsWith('repos/'+repository+'/pulls/'))return json(pulls.find(pull=>pull.number===Number(route.split('/').at(-1))));
    throw Error('Unexpected GitHub protocol request: '+route);
  }
  return {github:new GitHub('protocol-test-token',transport),storage:new Storage(),refs,commits,issues,pulls,counts,setLose(value){lose=value;},revoke(){push=false;}};
}
async function edit(context,state,source){const pages=await context.github.pages(state.head),page=pages.find(page=>page.source===source),passage=passages(page.markdown,source,page.blob).find(item=>item.type==='paragraph');return {workspace:state.id,expectedHead:state.head,operation:crypto.randomUUID(),patches:[{...passage,newMarkdown:'Reviewed '+passage.markdown,draftId:crypto.randomUUID()}]};}
for(const lost of ['ref','commit','issue','pr'])test('recover a lost '+lost+' response, commit another page and open exactly one cumulative PR',async()=>{
  const context=await protocol();let state=await workspace(context.storage,context.github,user),first=await edit(context,state,'docs/Developer/README.md');
  if(lost==='ref'||lost==='commit'){context.setLose(lost);await assert.rejects(()=>commit(context.storage,context.github,user,first,state),/Response lost/);state=await workspace(context.storage,context.github,user);}
  state=await commit(context.storage,context.github,user,first,state);const firstHead=state.head;
  // Losing the successful HTTP reply after the server receipt is recorded is also idempotent.
  state=await commit(context.storage,context.github,user,first,state);assert.equal(state.head,firstHead);
  const second=await edit(context,state,'docs/Developer/Documentation-System.md');state=await commit(context.storage,context.github,user,second,state);assert.notEqual(state.head,firstHead);
  const payload={workspace:state.id,operation:crypto.randomUUID()};if(lost==='issue'||lost==='pr'){context.setLose(lost);await assert.rejects(()=>openPull(context.storage,context.github,user,payload,state),/Response lost/);state=await workspace(context.storage,context.github,user);}
  const result=await openPull(context.storage,context.github,user,payload,state);assert.equal(result.number,200);assert.deepEqual(context.counts,{ref:1,commit:2,issue:1,pr:1});
  const changed=[...context.commits.get(state.head).files].filter(([source,text])=>context.commits.get(state.base).files.get(source)!==text).map(([source])=>source);assert.deepEqual(changed.sort(),['docs/Developer/Documentation-System.md','docs/Developer/README.md']);assert.match(context.pulls[0].body,/Closes #100/);
  assert.equal((await openPull(context.storage,context.github,user,payload,await workspace(context.storage,context.github,user))).number,200);
});
test('merged/deleted deliveries rotate; foreign PRs and revoked writers are refused',async()=>{
  const context=await protocol();await context.github.writer();let state=await workspace(context.storage,context.github,user);state=await commit(context.storage,context.github,user,await edit(context,state,'docs/Developer/README.md'),state);await openPull(context.storage,context.github,user,{workspace:state.id,operation:crypto.randomUUID()},state);
  context.pulls[0].state='closed';context.pulls[0].merged=true;context.refs.set('main',state.head);context.refs.delete(state.branch);const fresh=await workspace(context.storage,context.github,user);assert.notEqual(fresh.id,state.id);assert.equal(fresh.base,state.head);assert.equal(fresh.created,false);
  context.pulls[0].state='open';context.pulls[0].user={id:2,login:'other'};await assert.rejects(()=>workspace(new Storage(),context.github,user,200),/not this writer/);context.revoke();await assert.rejects(()=>context.github.writer(),/Current Limen write permission/);
});
