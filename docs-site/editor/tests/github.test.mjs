import test from 'node:test';
import assert from 'node:assert/strict';
import {GitHub,workspace,commit,openPull,repository} from '../github.mjs';
import {passages} from '../passages.mjs';
import {protocol,Storage,user} from './protocol.mjs';
async function edit(context,state,source){const pages=await context.github.pages(state.head),page=pages.find(page=>page.source===source),passage=passages(page.markdown,source,page.blob).find(item=>item.type==='paragraph');return {workspace:state.id,expectedHead:state.head,operation:crypto.randomUUID(),patches:[{...passage,newMarkdown:'Reviewed '+passage.markdown,draftId:crypto.randomUUID()}]};}
test('the default transport preserves the Worker fetch receiver',async()=>{
  const context=await protocol(),original=globalThis.fetch;
  globalThis.fetch=function(url,options){assert.equal(this,globalThis);return context.transport(url,options);};
  try{assert.deepEqual(await new GitHub('protocol-test-token').writer(),user);}finally{globalThis.fetch=original;}
});
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


test('an invalid selected issue does not persist a poisoned operation',async()=>{
  const context=await protocol();let state=await workspace(context.storage,context.github,user);state=await commit(context.storage,context.github,user,await edit(context,state,'docs/Developer/README.md'),state);
  await assert.rejects(()=>openPull(context.storage,context.github,user,{workspace:state.id,operation:crypto.randomUUID(),issue:999},state),/open Limen issue/);
  const fresh=await workspace(context.storage,context.github,user);assert.equal(fresh.operation,undefined);const result=await openPull(context.storage,context.github,user,{workspace:state.id,operation:crypto.randomUUID()},fresh);assert.equal(result.number,200);
});
for(const choice of ['replace-selected','replace-auto','resume-selected','resume-auto'])test('a closed pending issue permits '+choice+' without losing committed prose',async()=>{
  const context=await protocol();let state=await workspace(context.storage,context.github,user);state=await commit(context.storage,context.github,user,await edit(context,state,'docs/Developer/README.md'),state);
  const automatic=choice.endsWith('auto'),selected=automatic?null:await context.github.request('repos/'+repository+'/issues',{method:'POST',body:{title:'Revise documentation prose',body:'Maintain existing prose.'}}),payload={workspace:state.id,operation:crypto.randomUUID(),...(selected?{issue:selected.number}:{})},transport=context.github.transport;
  context.github.transport=async(url,options)=>{if(new URL(url).pathname==='/repos/'+repository+'/pulls'&&options.method==='POST')throw Error('Provider unavailable before PR creation');return transport(url,options);};
  await assert.rejects(()=>openPull(context.storage,context.github,user,payload,state),/Provider unavailable/);context.github.transport=transport;context.issues[0].state='closed';state=await workspace(context.storage,context.github,user);
  if(choice.startsWith('replace')){const replacement=await context.github.request('repos/'+repository+'/issues',{method:'POST',body:{title:'Continue the prose revision',body:'Retain the existing branch commits.'}});await openPull(context.storage,context.github,user,{...payload,issue:replacement.number},state);assert.match(context.pulls[0].body,new RegExp('Closes #'+replacement.number));}
  else{await assert.rejects(()=>openPull(context.storage,context.github,user,{workspace:state.id,operation:payload.operation},state),/tracking issue closed/);state=await workspace(context.storage,context.github,user);assert.equal(state.operation,undefined);if(automatic){await assert.rejects(()=>openPull(context.storage,context.github,user,{workspace:state.id,operation:crypto.randomUUID()},state),/tracking issue closed/);state=await workspace(context.storage,context.github,user);assert.equal(state.operation,undefined);}}
  state=await workspace(context.storage,context.github,user);state=await commit(context.storage,context.github,user,await edit(context,state,'docs/Developer/Documentation-System.md'),state);assert.equal(context.counts.commit,2);assert.equal(state.operation,undefined);
});
for(const merged of [false,true])test('a lost PR response followed by '+(merged?'merge and branch deletion':'closure')+' rotates the delivery',async()=>{
  const context=await protocol();let state=await workspace(context.storage,context.github,user);state=await commit(context.storage,context.github,user,await edit(context,state,'docs/Developer/README.md'),state);
  context.setLose('pr');await assert.rejects(()=>openPull(context.storage,context.github,user,{workspace:state.id,operation:crypto.randomUUID()},state),/Response lost/);
  context.pulls[0].state='closed';context.pulls[0].merged=merged;if(merged){context.refs.set('main',state.head);context.refs.delete(state.branch);}
  const fresh=await workspace(context.storage,context.github,user);assert.notEqual(fresh.id,state.id);assert.equal(fresh.operation,undefined);assert.equal(fresh.created,false);assert.equal(context.counts.pr,1);
});
test('a concurrent origin advance releases the pending operation for refresh and another commit',async()=>{
  const context=await protocol();let state=await workspace(context.storage,context.github,user);state=await commit(context.storage,context.github,user,await edit(context,state,'docs/Developer/README.md'),state);
  const payload=await edit(context,state,'docs/Developer/Documentation-System.md'),transport=context.github.transport,advanced='b'.repeat(40);let moved=false;
  context.github.transport=async(url,options)=>{if(!moved&&url.includes('/git/ref/heads/')&&(await context.storage.get('workspace')).operation){moved=true;context.commits.set(advanced,{...context.commits.get(state.head),parents:[state.head],message:'docs: update another passage'});context.refs.set(state.branch,advanced);}return transport(url,options);};
  await assert.rejects(()=>commit(context.storage,context.github,user,payload,state),/origin branch advanced/);state=await workspace(context.storage,context.github,user);assert.equal(state.operation,undefined);assert.equal(state.head,advanced);
  state=await commit(context.storage,context.github,user,await edit(context,state,'docs/Developer/Documentation-System.md'),state);assert.equal(context.counts.commit,2);assert.notEqual(state.head,advanced);
});
for(const applied of [false,true])test('a merged delivery reconciles a pending '+(applied?'committed':'unapplied')+' operation before rotating',async()=>{
  const context=await protocol();let state=await workspace(context.storage,context.github,user);state=await commit(context.storage,context.github,user,await edit(context,state,'docs/Developer/README.md'),state);await openPull(context.storage,context.github,user,{workspace:state.id,operation:crypto.randomUUID()},state);
  const payload=await edit(context,state,'docs/Developer/Documentation-System.md'),transport=context.github.transport;
  if(applied)context.setLose('commit');else context.github.transport=async(url,options)=>{if(url.endsWith('/graphql')&&JSON.parse(options.body).variables.input)throw Error('Response lost before commit');return transport(url,options);};
  await assert.rejects(()=>commit(context.storage,context.github,user,payload,state),/Response lost/);context.github.transport=transport;
  const head=context.refs.get(state.branch);context.pulls[0].state='closed';context.pulls[0].merged=true;context.pulls[0].head.sha=head;context.refs.set('main',head);context.refs.delete(state.branch);
  state=await workspace(context.storage,context.github,user);assert.equal(state.operation.kind,'commit');
  if(applied){state=await commit(context.storage,context.github,user,payload,state);assert.equal(state.receipt.head,head);}else await assert.rejects(()=>commit(context.storage,context.github,user,payload,state),/delivery closed/);
  const fresh=await workspace(context.storage,context.github,user);assert.notEqual(fresh.id,state.id);assert.equal(fresh.base,head);assert.equal(context.counts.commit,applied?2:1);
});
for(const applied of [false,true])test('reopening a delivery clears a pending closed-head snapshot and permits cumulative commits, applied='+applied,async()=>{
  const context=await protocol();let state=await workspace(context.storage,context.github,user);state=await commit(context.storage,context.github,user,await edit(context,state,'docs/Developer/README.md'),state);await openPull(context.storage,context.github,user,{workspace:state.id,operation:crypto.randomUUID()},state);
  const payload=await edit(context,state,'docs/Developer/Documentation-System.md'),transport=context.github.transport;
  if(applied)context.setLose('commit');else context.github.transport=async(url,options)=>{if(url.endsWith('/graphql')&&JSON.parse(options.body).variables.input)throw Error('Response lost before commit');return transport(url,options);};
  await assert.rejects(()=>commit(context.storage,context.github,user,payload,state),/Response lost/);context.github.transport=transport;
  context.pulls[0].state='closed';context.pulls[0].head.sha=context.refs.get(state.branch);state=await workspace(context.storage,context.github,user);assert.equal(state.closedHead,context.refs.get(state.branch));
  context.pulls[0].state='open';state=await workspace(context.storage,context.github,user);assert.equal(state.closedHead,undefined);state=await commit(context.storage,context.github,user,payload,state);assert.equal(state.operation,undefined);
  state=await workspace(context.storage,context.github,user);state=await commit(context.storage,context.github,user,await edit(context,state,'docs/Developer/README.md'),state);assert.equal(context.counts.commit,3);assert.equal(context.counts.pr,1);
});
