import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import CONTENT from '../../reader/content.json' with {type:'json'};
import {GitHub,repository} from '../github.mjs';
export const user={id:1,login:'writer'},hash=value=>createHash('sha1').update(value).digest('hex');
export class Storage{constructor(){this.values=new Map();}async get(key){return structuredClone(this.values.get(key));}async put(key,value){this.values.set(key,structuredClone(value));}async delete(key){return this.values.delete(key);}async setAlarm(value){this.alarm=value;}async getAlarm(){return this.alarm;}async list(){return structuredClone(this.values);}}
export async function protocol(){
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
    if(route.startsWith('repos/'+repository+'/issues/')){const issue=issues.find(issue=>issue.number===Number(route.split('/').at(-1)));return issue?json(issue):json({message:'Not found'},404);}
    if(route==='repos/'+repository+'/pulls'){
      if(options.method==='POST'){const pull={...body,number:200+pulls.length,state:'open',user,head:{ref:body.head,sha:refs.get(body.head),repo:{full_name:repository}},base:{ref:body.base,sha:refs.get('main'),repo:{full_name:repository}},html_url:'https://github.com/Vaquum/Limen/pull/'+(200+pulls.length)};pulls.push(pull);counts.pr++;if(lose==='pr'){lose=null;throw Error('Response lost after PR');}return json(pull,201);}return json(pulls.filter(pull=>'Vaquum:'+pull.head.ref===query.get('head')));
    }
    if(route.startsWith('repos/'+repository+'/pulls/'))return json(pulls.find(pull=>pull.number===Number(route.split('/').at(-1))));
    throw Error('Unexpected GitHub protocol request: '+route);
  }
  return {transport,github:new GitHub('protocol-test-token',transport),storage:new Storage(),refs,commits,issues,pulls,counts,setLose(value){lose=value;},revoke(){push=false;}};
}
