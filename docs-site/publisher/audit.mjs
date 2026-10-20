import {spawnSync} from 'node:child_process';
import {mkdirSync,writeFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {inspectAudit} from '../scripts/audit-security.mjs';

const publisher=path.dirname(fileURLToPath(import.meta.url));
const result=spawnSync('npm',['audit','--json'],{cwd:publisher,encoding:'utf8',timeout:60000});
const output=path.resolve(publisher,'../../.docs-evidence');
mkdirSync(output,{recursive:true});
writeFileSync(path.join(output,'publisher-npm-audit.json'),result.stdout||'');
if(inspectAudit(result)!==0)throw Error('The deployment toolchain has no advisory exceptions');
console.log('No publisher npm vulnerabilities found');
