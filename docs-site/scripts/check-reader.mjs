import {spawnSync,execFileSync} from 'node:child_process';
import fs from 'node:fs/promises';
import {gzipSync} from 'node:zlib';
import path from 'node:path';
import {parseArgs} from 'node:util';
import {hash} from '../reader/render.mjs';
import {withServer} from './proof-server.mjs';
const root=path.resolve(process.cwd(),'..');
const {values}=parseArgs({options:{evidence:{type:'string',default:'../.docs-evidence/current.json'},head:{type:'string'},tested:{type:'string'},tree:{type:'string'}}});
const evidencePath=path.resolve(values.evidence);
const evidence=JSON.parse(await fs.readFile(evidencePath,'utf8'));
if(evidence.validation?.status!=='passed')throw Error('Current successful research execution is required before reader checks');
const built=JSON.parse(await fs.readFile('reader-build/build.json','utf8'));
const head=values.head||process.env.LIMEN_EXPECTED_HEAD||execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim();
const tested=execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim();
const tree=execFileSync('git',['rev-parse','HEAD^{tree}'],{cwd:root,encoding:'utf8'}).trim();
if((values.tested&&values.tested!==tested)||(values.tree&&values.tree!==tree))throw Error('Expected checkout identity differs from the actual source');
if(built.head!==head||built.tested_commit!==tested||built.tree!==tree||built.evidence_sha256!==hash(await fs.readFile(evidencePath)))throw Error('Reader build does not match the current tested revision');
for(const [name,expected] of Object.entries(built.files))if(hash(await fs.readFile(path.join('reader-build',name)))!==expected)throw Error('Built reader member changed: '+name);
let js=0,css=0;for(const name of await fs.readdir('reader-build/assets')){if(name.endsWith('.mjs'))js+=gzipSync(await fs.readFile(path.join('reader-build/assets',name))).length;if(name.endsWith('.css'))css+=gzipSync(await fs.readFile(path.join('reader-build/assets',name))).length;}
if(js>60*1024||css>24*1024)throw Error(`Reader asset budget exceeded: JS ${js}, CSS ${css}`);
const started=performance.now();
await withServer('scripts/serve-reader.mjs',{READER_PORT:'0',READER_TEST_FIXTURES:'1'},port=>withServer('editor/tests/serve.mjs',{EDITOR_TEST_PORT:'0'},editorPort=>{
  const result=spawnSync(process.execPath,['node_modules/@playwright/test/cli.js','test'],{stdio:'inherit',timeout:180000,env:{...process.env,READER_PORT:String(port),READER_SERVER_MANAGED:'1',EDITOR_TEST_ORIGIN:'https://127.0.0.1:'+editorPort,EDITOR_SERVER_MANAGED:'1'}});
  if(result.error)throw result.error;if(result.status!==0)throw Error(`Reader browser proof failed: ${result.status}`);
  console.log(`Reader proof passed in ${((performance.now()-started)/1000).toFixed(2)}s; JS ${js}B gzip, CSS ${css}B gzip.`);
}));
