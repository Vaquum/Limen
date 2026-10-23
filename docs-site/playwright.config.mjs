import {defineConfig} from '@playwright/test';
const port=process.env.READER_PORT||'4331',origin='http://127.0.0.1:'+port;
export default defineConfig({
  testDir:'./tests',testMatch:['book.spec.mjs','platform.spec.mjs'],timeout:30000,expect:{timeout:6000},
  fullyParallel:false,workers:1,retries:0,
  reporter:[['list'],['json',{outputFile:'../.docs-evidence/browser/results.json'}]],
  outputDir:'../.docs-evidence/browser/test-results',
  use:{baseURL:origin,headless:true,screenshot:'only-on-failure',trace:'retain-on-failure'},
  projects:[{name:'desktop',use:{viewport:{width:1440,height:1000}}},{name:'phone',use:{viewport:{width:390,height:844}}},{name:'webkit-phone',testMatch:'platform.spec.mjs',use:{browserName:'webkit',viewport:{width:390,height:844}}}],
  webServer:process.env.READER_SERVER_MANAGED==='1'?undefined:{command:'node scripts/serve-reader.mjs',url:origin+'/limen/',env:{READER_PORT:port,READER_TEST_FIXTURES:'1'},reuseExistingServer:false,timeout:15000}
});
