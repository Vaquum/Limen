import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {auditDependencies} from '../scripts/audit-security.mjs';

const publisher=path.dirname(fileURLToPath(import.meta.url));
const output=path.resolve(publisher,'../../.docs-evidence');
if(await auditDependencies(publisher,output,{stem:'publisher-npm-audit'})!==0)throw Error('The deployment toolchain has no advisory exceptions');
console.log('No publisher npm vulnerabilities found');
