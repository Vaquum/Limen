import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const site=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),out=path.join(site,'editor-build');
await fs.rm(out,{recursive:true,force:true});await fs.mkdir(path.join(out,'limen/edit/assets'),{recursive:true});await fs.mkdir(path.join(out,'limen/assets/fonts'),{recursive:true});
for(const name of await fs.readdir(path.join(site,'editor/assets')))await fs.copyFile(path.join(site,'editor/assets',name),path.join(out,name==='index.html'?'limen/edit/index.html':'limen/edit/assets/'+name));
for(const name of ['book.css','main.mjs','code.mjs','place.mjs','navigation.mjs','search.mjs','interfaces.mjs','motion.mjs'])await fs.copyFile(path.join(site,'reader',name),path.join(out,'limen/assets',name));
for(const [from,to] of [['@fontsource-variable/archivo/files/archivo-latin-standard-normal.woff2','archivo.woff2'],['@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-400-normal.woff2','plex.woff2'],['@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-500-normal.woff2','plex-medium.woff2'],['@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-400-italic.woff2','plex-italic.woff2'],['@fontsource-variable/archivo/LICENSE','archivo-OFL.txt'],['@fontsource/ibm-plex-mono/LICENSE','plex-OFL.txt']])await fs.copyFile(path.join(site,'node_modules',from),path.join(out,'limen/assets/fonts',to));
console.log('Authoring assets assembled independently of the public release.');
