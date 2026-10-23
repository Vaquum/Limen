import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import MarkdownIt from 'markdown-it';
import {headingIdentities} from '../../reader/headings.mjs';
import {renderMarkdown} from '../../reader/render.mjs';
import {searchEntries} from '../../reader/search-index.mjs';
import {compatibilityMarkdown,compatibilityContents} from '../../scripts/render-compat-input.mjs';
const vectors=JSON.parse(await fs.readFile(new URL('./heading-vectors.json',import.meta.url),'utf8'));
test('escaped heading punctuation has decoded contents labels without changing destinations or body bytes',()=>{
  const markdown='# Research\n\n## Heading\\. \\[literal\\] **bold** `code`\n',page={source:'docs/Developer/README.md',markdown,title:'Research',url:'/limen/developer/home/',part:'Developer'},evidence={revision:{head:'a'.repeat(40)},sources:[],displays:[],validation:{scenarios:[]},package:{version:'5.46.2'}};
  const rendered=renderMarkdown(page,[page],evidence);assert.equal(rendered.headings[0].title,'Heading. [literal] bold code');
  const legacy='<p>Body\\.</p><li><a href="#heading-literal-bold-code">Heading\\. \\[literal\\] **bold** code</a></li>';
  assert.equal(compatibilityContents(legacy,markdown),'<p>Body\\.</p><li><a href="#heading-literal-bold-code">Heading. [literal] bold code</a></li>');
});
test('compatibility uses current excerpts and removes only generated heading annotations outside code',()=>{
  const anchor='<a id="retained" data-reader-heading="compiled"></a>',source=anchor+'\n\n## Revised heading\n\n```html\n'+anchor+'\n```\n\nCurrent **prose** and `source`.\n';
  assert.equal(compatibilityMarkdown(source),source.slice(anchor.length+1));
  for(const vector of vectors){const before=new MarkdownIt({html:true}).parse(vector.markdown,{}),after=new MarkdownIt({html:true}).parse(compatibilityMarkdown(vector.markdown),{});assert.deepEqual(after.filter(token=>token.type==='heading_open').map(token=>token.tag),before.filter(token=>token.type==='heading_open').map(token=>token.tag));}
});
test('renderer, source-fragment rule and search retain the shared heading identities',()=>{
  for(const vector of vectors){
    const identities=[...headingIdentities(new MarkdownIt({html:true}).parse(vector.markdown,{})).values()];assert.deepEqual(identities.map(item=>item.id),vector.ids);assert.deepEqual(identities.map(item=>item.kind),vector.kinds);
    const page={source:'docs/Developer/README.md',markdown:vector.markdown,title:'Research',url:'/limen/developer/home/',part:'Developer'},evidence={revision:{head:'a'.repeat(40)},sources:[],displays:[],validation:{scenarios:[]},package:{version:'5.46.2'}};
    const rendered=renderMarkdown(page,[page],evidence);assert.deepEqual(rendered.headings.map(item=>item.id),vector.ids.slice(1));
    const indexed=searchEntries(page).filter(item=>item.route.includes('#'));assert.deepEqual(indexed.map(item=>item.route.split('#')[1]),vector.ids.slice(1));
    for(const id of vector.ids.slice(1))assert.ok(rendered.html.includes('id="'+id+'"'));
  }
});
