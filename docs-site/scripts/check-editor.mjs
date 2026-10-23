import {spawnSync} from 'node:child_process';
import {withServer} from './proof-server.mjs';
await withServer('editor/tests/serve.mjs',{EDITOR_TEST_PORT:'0'},port=>{
  const result=spawnSync(process.execPath,['node_modules/@playwright/test/cli.js','test','--config','playwright.editor.config.mjs'],{stdio:'inherit',timeout:180000,env:{...process.env,EDITOR_TEST_ORIGIN:'https://127.0.0.1:'+port,EDITOR_SERVER_MANAGED:'1'}});
  if(result.error)throw result.error;if(result.status!==0)throw Error('Authoring browser proof failed: '+result.status);
});
