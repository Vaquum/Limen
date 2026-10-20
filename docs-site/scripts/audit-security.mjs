import {spawnSync} from 'node:child_process';
import path from 'node:path';
import {mkdirSync,writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';

// Tracked 2026-10-04: braces <=3.0.3 has no patched release. Repository-owned
// patterns produce static output. Retire when a patched release resolves.
const TRACKED_ADVISORIES=new Set(['https://github.com/advisories/GHSA-vfj7-8cjw-p6xm']);
const object=value=>value!==null&&typeof value==='object'&&!Array.isArray(value);
export function inspectAudit(result){
  if(result.error||result.signal||![0,1].includes(result.status))throw Error(`npm audit failed: ${result.error?.message||result.signal||result.stderr||result.status}`);
  if(typeof result.stdout!=='string'||!result.stdout.trim())throw Error('npm audit produced no JSON report');
  const report=JSON.parse(result.stdout);
  if(!object(report)||report.error||report.auditReportVersion!==2||!object(report.vulnerabilities)||!object(report.metadata?.vulnerabilities))throw Error('npm audit did not return a complete vulnerability report');
  const entries=report.vulnerabilities,count=Object.keys(entries).length,counts=report.metadata.vulnerabilities;
  for(const field of ['info','low','moderate','high','critical','total'])if(!Number.isInteger(counts[field])||counts[field]<0)throw Error(`Invalid audit count: ${field}`);
  if(counts.total!==count||counts.total!==['info','low','moderate','high','critical'].reduce((sum,key)=>sum+counts[key],0))throw Error('Audit package totals disagree');
  if((count===0)!==(result.status===0))throw Error('Audit status disagrees with its vulnerability report');
  function advisories(name,visiting=new Set()){
    if(visiting.has(name))return new Set();
    const item=entries[name];if(!object(item)||!Array.isArray(item.via)||!item.via.length)throw Error(`Missing advisory paths: ${name}`);
    const next=new Set(visiting).add(name),urls=new Set();
    for(const via of item.via){
      if(typeof via==='string'){if(!Object.hasOwn(entries,via))throw Error(`Unknown advisory dependency: ${via}`);for(const url of advisories(via,next))urls.add(url);}
      else {if(!object(via)||typeof via.url!=='string'||!via.url.startsWith('https://github.com/advisories/'))throw Error(`Missing advisory identity: ${name}`);urls.add(via.url);}
    }
    return urls;
  }
  const untracked=[];
  for(const name of Object.keys(entries)){const urls=advisories(name);if(!urls.size)throw Error(`No advisory identity: ${name}`);for(const url of urls)if(!TRACKED_ADVISORIES.has(url))untracked.push(`${name}: ${url}`);}
  if(untracked.length)throw Error(`Untracked docs-site npm vulnerabilities: ${untracked.join(', ')}`);
  return count;
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  try{const site=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');const result=spawnSync('npm',['audit','--json'],{cwd:site,encoding:'utf8',timeout:60000});const artifacts=path.resolve(site,'../.docs-evidence');mkdirSync(artifacts,{recursive:true});writeFileSync(path.join(artifacts,'npm-audit.json'),result.stdout||'');const count=inspectAudit(result);console.log(count?`Tracked docs-site npm vulnerabilities: ${count}`:'No docs-site npm vulnerabilities found');}
  catch(error){console.error(error.message);process.exitCode=1;}
}
