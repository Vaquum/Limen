import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {inspectAudit} from '../scripts/audit-security.mjs';

// Retained registry report from the removed renderer; its former exception must fail.
const retained=JSON.parse(fs.readFileSync(new URL('./fixtures/npm-audit.json',import.meta.url),'utf8'));
const result=report=>({status:1,stdout:JSON.stringify(report),stderr:''});
test('the removed renderer advisory is no longer exempt',()=>{
  assert.throws(()=>inspectAudit(result(retained)),/Untracked/);
});
test('the current complete graph has no vulnerabilities',()=>{
  const current=JSON.parse(fs.readFileSync(new URL('./fixtures/npm-audit-current.json',import.meta.url),'utf8'));
  assert.equal(inspectAudit({status:0,stdout:JSON.stringify(current),stderr:''}),0);
});
test('transport, missing data and malformed reports fail closed',()=>{
  for(const value of [{status:1,stdout:'',stderr:'network failed'}, {status:null,error:Error('timeout')}, {status:0,stdout:'not json'},result({}),result({...retained,error:{code:'EAI_AGAIN'}})])assert.throws(()=>inspectAudit(value));
  for(const mutate of [r=>delete r.metadata,r=>delete r.metadata.vulnerabilities.total,r=>r.metadata.vulnerabilities.total++,r=>r.metadata.vulnerabilities.high=-1]){
    const report=structuredClone(retained);mutate(report);assert.throws(()=>inspectAudit(result(report)));
  }
});
test('a dependency path does not excuse a second advisory',()=>{
  const report=structuredClone(retained);
  const parent=Object.values(report.vulnerabilities).find(item=>item.via.some(via=>typeof via==='string'));
  parent.via.push({url:'https://github.com/advisories/GHSA-xxxx-xxxx-xxxx'});
  assert.throws(()=>inspectAudit(result(report)),/Untracked/);
});
test('missing dependencies and identity-free cycles cannot pass',()=>{
  const report=structuredClone(retained),names=Object.keys(report.vulnerabilities);
  report.vulnerabilities[names[0]].via=['missing-dependency'];
  assert.throws(()=>inspectAudit(result(report)),/Unknown advisory/);
  report.vulnerabilities[names[0]].via=[names[1]];report.vulnerabilities[names[1]].via=[names[0]];
  assert.throws(()=>inspectAudit(result(report)),/No advisory identity/);
});
test('exit status must agree with the actual report',()=>{
  assert.throws(()=>inspectAudit({...result(retained),status:0}),/status disagrees/);
});
