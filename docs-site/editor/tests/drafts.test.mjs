import test from 'node:test';
import assert from 'node:assert/strict';
import {draftStore} from '../assets/drafts.mjs';
import {inlinePreview} from '../preview.mjs';
class BrowserStorage{constructor(){this.values=new Map();}get length(){return this.values.size;}key(index){return [...this.values.keys()][index];}getItem(key){return this.values.get(key)||null;}setItem(key,value){this.values.set(key,value);}}
const patch={path:'docs/Developer/README.md',baseBlobSha:'a'.repeat(40),sourceRange:{start:10,end:20},originalText:'Developers',newMarkdown:'Contributors'};
test('stale tabs retain competing prose; acknowledging a commit cannot clear a newer version',()=>{
  const storage=new BrowserStorage(),first=draftStore(1,'delivery',storage),second=draftStore(1,'delivery',storage);
  const a=first.save(patch,[]),b=second.save({...patch,newMarkdown:'Maintainers'},[]);assert.equal(first.conflicts(),true);assert.equal(first.heads().length,2);
  second.undo(b.id);const newer=first.save({...patch,newMarkdown:'Package contributors'},[a.id]);first.confirmed([a.id]);assert.deepEqual(first.heads().map(item=>item.id),[newer.id]);assert.equal(first.conflicts(),false);
  assert.equal(draftStore(2,'delivery',storage).heads().length,0);assert.equal(draftStore(1,'other delivery',storage).heads().length,0);
});
test('restored drafts retain source-relative links and authoring navigation',()=>{
  const page={source:'docs/Historical-Data.md',markdown:'[Source reference](Historical-Data-Reference.md)'},target={source:'docs/Historical-Data-Reference.md',route:'/reference/historical-data/',url:'/limen/reference/historical-data/'};
  const html=inlinePreview(page.markdown,page,[target],41);assert.match(html,/href="\/limen\/edit\/view\/reference\/historical-data\/\?pr=41"/);assert.match(html,/data-source-href="Historical-Data-Reference.md"/);
});
