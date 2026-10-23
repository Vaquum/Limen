import {EditError,applyPatches} from './passages.mjs';
import {documentTitle} from '../reader/headings.mjs';
import CONTENT from '../reader/content.json' with {type:'json'};

export const repository='Vaquum/Limen';
export class GitHub{
  constructor(token,transport=(url,options)=>globalThis.fetch(url,options)){this.token=token;this.transport=transport;}
  async request(path,{method='GET',body,missing=false}={}){
    const response=await this.transport('https://api.github.com/'+path,{method,headers:{Authorization:'Bearer '+this.token,Accept:'application/vnd.github+json','X-GitHub-Api-Version':'2026-03-10','User-Agent':'Limen-docs-editor',...(body?{'Content-Type':'application/json'}:{})},...(body?{body:JSON.stringify(body)}:{})});
    if(missing&&response.status===404)return null;
    if(!response.ok)throw new EditError('GitHub '+response.status+': '+(await response.text()).slice(0,500),response.status===401||response.status===403?response.status:502);
    return response.status===204?null:response.json();
  }
  async writer(){const user=await this.request('user'),repo=await this.request('repos/'+repository);if(repo.permissions?.push!==true)throw new EditError('Current Limen write permission is required',403);return {id:user.id,login:user.login};}
  async head(branch,missing=false){return (await this.request('repos/'+repository+'/git/ref/heads/'+encodeURIComponent(branch),{missing}))?.object.sha;}
  async pages(head){
    const paths=Object.keys(CONTENT.pages),variables={};const fields=paths.map((path,index)=>{variables['p'+index]=head+':'+path;return `p${index}:object(expression:$p${index}){... on Blob{oid text}}`;});
    const data=await this.request('graphql',{method:'POST',body:{query:`query(${paths.map((_,i)=>'$p'+i+':String!').join(',')}){repository(owner:"Vaquum",name:"Limen"){${fields.join(' ')}}}`,variables}});
    if(data.errors)throw new EditError('GitHub source read failed: '+data.errors.map(item=>item.message).join('; '),502);
    return paths.map((source,index)=>{const blob=data.data.repository['p'+index];if(!blob?.text)throw new EditError('Source unavailable: '+source,409);return {source,...CONTENT.pages[source],markdown:blob.text,blob:blob.oid,title:documentTitle(blob.text)||CONTENT.product.name,url:CONTENT.product.basePath+CONTENT.pages[source].route};});
  }
  async pull(number){return this.request('repos/'+repository+'/pulls/'+number);}
  async pulls(branch){return this.request('repos/'+repository+'/pulls?state=all&base=main&head='+encodeURIComponent('Vaquum:'+branch)+'&per_page=100');}
  async issueByMarker(login,marker){
    for(let page=1;;page++){
      const issues=await this.request('repos/'+repository+'/issues?state=all&creator='+encodeURIComponent(login)+'&per_page=100&page='+page);
      const matches=issues.filter(issue=>!issue.pull_request&&issue.body?.includes(marker));
      if(matches.length>1)throw new EditError('More than one issue claims this editing workspace',409);
      if(matches.length)return matches[0];if(issues.length<100)return null;
    }
  }
}
const assert=(condition,message)=>{if(!condition)throw new EditError(message,409);};
function ownPull(pull,user,branch){
  assert(pull.user.id===user.id&&pull.head.repo?.full_name===repository&&pull.head.ref===branch&&pull.base.repo.full_name===repository&&pull.base.ref==='main','PR is not this writer’s Limen delivery');
}
export async function workspace(storage,github,user,continuing){
  let state=await storage.get('workspace');
  if(continuing){
    const pull=await github.pull(continuing);ownPull(pull,user,pull.head.ref);assert(pull.state==='open','Only an open PR can be continued');
    if(!state)state={id:crypto.randomUUID(),branch:pull.head.ref,head:pull.head.sha,base:pull.base.sha,created:true,pr:pull.number,continuing:true,files:[]};
    assert(state.branch===pull.head.ref,'Workspace does not match the selected PR');
  }
  if(state?.operation?.kind==='pr'&&!state.pr){
    const existing=await github.pulls(state.branch);assert(existing.length<=1,'Multiple PRs claim this delivery');
    if(existing.length){const pull=existing[0];ownPull(pull,user,state.branch);assert(pull.body?.includes(`<!-- limen-editor workspace=${state.id} -->`),'PR does not identify this editing workspace');state.pr=pull.number;delete state.operation;}
  }
  if(state?.pr){
    const pull=await github.pull(state.pr);ownPull(pull,user,state.branch);
    if(pull.state!=='open'){if(state.operation)state.closedHead=pull.head.sha;else state=null;}
    else delete state.closedHead;
  }
  if(!state){const head=await github.head('main'),id=crypto.randomUUID();state={id,branch:`codex/docs-edit-${user.id}-${id}`,base:head,head,created:false,files:[]};}
  if(state.created&&!state.closedHead){const head=await github.head(state.branch,true);assert(head,'The unmerged branch is missing; recover it in GitHub before continuing');state.remoteHead=head;if(!state.operation)state.head=head;}
  await storage.put('workspace',state);return state;
}
async function commitMatches(github,state,operation,head){
  if(head===operation.expectedHead)return false;
  const commit=await github.request('repos/'+repository+'/git/commits/'+head);
  if(commit.parents.length!==1||commit.parents[0].sha!==operation.expectedHead||!commit.message.includes('Limen-Edit-Operation: '+operation.id))return false;
  const pages=await github.pages(head);
  return operation.files.every(file=>pages.find(page=>page.source===file.path)?.markdown===file.text);
}
export async function commit(storage,github,user,payload,state){
  assert(payload.workspace===state.id,'This draft belongs to another delivery');
  if(state.receipts?.[payload.operation])return {...state,receipt:state.receipts[payload.operation]};
  if(!state.operation){
    assert(payload.expectedHead===state.head&&(!state.remoteHead||state.remoteHead===state.head),'The origin branch moved; refresh and reconcile drafts');
    assert(Array.isArray(payload.patches)&&payload.patches.length>0&&payload.patches.length<=512,'Commit requires 1–512 passages');
    const pages=await github.pages(state.head),files=[];
    for(const path of new Set(payload.patches.map(patch=>patch.path))){
      assert(Object.hasOwn(CONTENT.pages,path),'This source is outside the deployed editor allowlist');
      const page=pages.find(item=>item.source===path),patches=payload.patches.filter(patch=>patch.path===path);
      const text=applyPatches(page.markdown,path,page.blob,patches);if(text!==page.markdown)files.push({path,text});
    }
    assert(files.length,'There are no changed passages to commit');
    state.operation={kind:'commit',id:payload.operation,expectedHead:state.head,files,drafts:payload.patches.map(patch=>patch.draftId)};
    await storage.put('workspace',state);
  }
  const operation=state.operation;assert(operation.kind==='commit'&&operation.id===payload.operation,'Resume the existing operation before another commit');
  let head=state.closedHead||await github.head(state.branch,true);
  if(state.closedHead){if(!await commitMatches(github,state,operation,head)){delete state.operation;await storage.put('workspace',state);throw new EditError('The delivery closed before this commit; Refresh to recover retained drafts in another delivery',409);}}
  else{
  if(!head){assert(!state.created,'The delivery branch was removed');await github.request('repos/'+repository+'/git/refs',{method:'POST',body:{ref:'refs/heads/'+state.branch,sha:state.base}});head=state.base;}
  state.created=true;await storage.put('workspace',state);
  if(head!==operation.expectedHead){if(!await commitMatches(github,state,operation,head)){state.head=head;state.remoteHead=head;delete state.operation;await storage.put('workspace',state);throw new EditError('The origin branch advanced; Refresh to reconcile the retained drafts',409);}}
  else{
    const response=await github.request('graphql',{method:'POST',body:{query:'mutation($input:CreateCommitOnBranchInput!){createCommitOnBranch(input:$input){commit{oid url}}}',variables:{input:{branch:{repositoryNameWithOwner:repository,branchName:state.branch},expectedHeadOid:operation.expectedHead,message:{headline:'docs: revise documentation prose',body:'Limen-Edit-Operation: '+operation.id},fileChanges:{additions:operation.files.map(file=>({path:file.path,contents:Buffer.from(file.text).toString('base64')}))}}}}});
    if(response.errors){delete state.operation;await storage.put('workspace',state);throw new EditError('Commit refused; Refresh to reconcile retained drafts: '+response.errors.map(item=>item.message).join('; '),409);}
    head=response.data.createCommitOnBranch.commit.oid;
    assert(await commitMatches(github,state,operation,head),'GitHub commit read-back differs from the submitted prose');
  }
  }
  const receipt={operation:operation.id,head,drafts:operation.drafts,files:operation.files.map(file=>file.path)};
  state.head=head;state.files=[...new Set([...state.files,...receipt.files])];state.receipt=receipt;state.receipts={...state.receipts,[operation.id]:receipt};delete state.operation;delete state.remoteHead;await storage.put('workspace',state);return {...state,receipt};
}
export async function openPull(storage,github,user,payload,state){
  assert(payload.workspace===state.id&&state.created,'Commit the prose before opening a PR');assert(!state.operation||state.operation.kind==='pr','Recover the pending commit first');
  const existing=await github.pulls(state.branch),open=existing.filter(pull=>pull.state==='open');
  assert(open.length<=1,'Multiple open PRs claim this delivery');
  if(open.length){ownPull(open[0],user,state.branch);state.pr=open[0].number;delete state.operation;await storage.put('workspace',state);return {number:open[0].number,url:open[0].html_url};}
  assert(!existing.length,'This delivery already has a closed PR; refresh to start another');
  const marker=`<!-- limen-editor workspace=${state.id} -->`;
  let selected=null;if(payload.issue){assert(Number.isSafeInteger(payload.issue)&&payload.issue>0,'Choose a valid issue number');selected=await github.request('repos/'+repository+'/issues/'+payload.issue,{missing:true});assert(selected&&!selected.pull_request&&selected.state==='open','Choose a genuine open Limen issue');}
  if(!state.operation){state.operation={kind:'pr',id:payload.operation,issue:selected?.number||null};await storage.put('workspace',state);}
  else if(selected&&state.operation.issue!==selected.number){state.operation.issue=selected.number;await storage.put('workspace',state);}
  let issue;
  if(state.operation.issue)issue=await github.request('repos/'+repository+'/issues/'+state.operation.issue);
  else{
    issue=await github.issueByMarker(user.login,marker);
    if(!issue)issue=await github.request('repos/'+repository+'/issues',{method:'POST',body:{title:'docs: revise documentation prose',body:'Revise existing prose in:\n\n'+state.files.map(path=>'- `'+path+'`').join('\n')+'\n\nPreserve examples and source contracts; validate through the normal documentation and package checks.\n\n'+marker}});
  }
  if(issue.pull_request||issue.state!=='open'){delete state.operation;await storage.put('workspace',state);throw new EditError('The tracking issue closed; choose an open Limen issue. Existing commits and drafts are retained.',409);}
  state.operation.issue=issue.number;await storage.put('workspace',state);
  const pull=await github.request('repos/'+repository+'/pulls',{method:'POST',body:{title:'docs: revise documentation prose',head:state.branch,base:'main',draft:false,body:'Revise existing documentation prose. The commits on this branch are cumulative; verification and review follow the normal repository gates. Public documentation changes with the next package release.\n\nCloses #'+issue.number+'\n\n'+marker}});
  ownPull(pull,user,state.branch);state.pr=pull.number;delete state.operation;await storage.put('workspace',state);return {number:pull.number,url:pull.html_url};
}
