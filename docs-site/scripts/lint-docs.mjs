import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {lint} from 'markdownlint/promise';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
async function markdown(directory,include){
  const files=[];
  for(const entry of await fs.readdir(directory,{withFileTypes:true})){
    const target=path.join(directory,entry.name);
    if(entry.isDirectory())files.push(...await markdown(target,include));
    else if(include(entry.name))files.push(target);
  }
  return files;
}
const files=[path.join(root,'README.md'),path.join(root,'CHANGELOG.md'),
  ...await markdown(path.join(root,'docs'),name=>name.endsWith('.md')),
  ...await markdown(path.join(root,'limen'),name=>name==='README.md')];
const config=JSON.parse(await fs.readFile(path.join(root,'.markdownlint.json'),'utf8'));
const results=await lint({files,config});
const findings=results.toString();
if(findings){console.error(findings);process.exitCode=1;}
else console.log(`Markdown lint passed for ${files.length} maintained documents.`);
