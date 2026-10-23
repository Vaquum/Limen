import {defineConfig} from '@playwright/test';
const port=process.env.READER_PORT||'4331',origin='http://127.0.0.1:'+port;
const editorOrigin=process.env.EDITOR_TEST_ORIGIN||'https://127.0.0.1:4335';
export default defineConfig({
  testDir:'./tests',testMatch:['book.spec.mjs','platform.spec.mjs','editor.spec.mjs'],timeout:30000,expect:{timeout:6000},
  fullyParallel:false,workers:1,retries:0,
  reporter:[['list'],['json',{outputFile:'../.docs-evidence/browser/results.json'}]],
  outputDir:'../.docs-evidence/browser/test-results',
  use:{baseURL:origin,headless:true,screenshot:'only-on-failure',trace:'retain-on-failure'},
  projects:[{name:'desktop',testMatch:['book.spec.mjs','platform.spec.mjs'],use:{viewport:{width:1440,height:1000}}},{name:'phone',testMatch:['book.spec.mjs','platform.spec.mjs'],use:{viewport:{width:390,height:844}}},{name:'webkit-phone',testMatch:'platform.spec.mjs',use:{browserName:'webkit',viewport:{width:390,height:844}}},{name:'editor-desktop',testMatch:'editor.spec.mjs',use:{baseURL:editorOrigin,ignoreHTTPSErrors:true,viewport:{width:1440,height:1000}}},{name:'editor-phone',testMatch:'editor.spec.mjs',use:{baseURL:editorOrigin,ignoreHTTPSErrors:true,viewport:{width:390,height:844},reducedMotion:'reduce'}}],
  webServer:[process.env.READER_SERVER_MANAGED==='1'?null:{command:'node scripts/serve-reader.mjs',url:origin+'/limen/',env:{READER_PORT:port,READER_TEST_FIXTURES:'1'},reuseExistingServer:false,timeout:15000},process.env.EDITOR_SERVER_MANAGED==='1'?null:{command:'node editor/tests/serve.mjs',url:editorOrigin+'/limen/edit/',ignoreHTTPSErrors:true,reuseExistingServer:false,timeout:30000}].filter(Boolean)
});
