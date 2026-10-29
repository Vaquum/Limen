import {test,expect} from '@playwright/test';
import fs from 'node:fs/promises';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
const root=path.resolve(process.cwd(),'..');
const failures=new WeakMap();
test.beforeEach(async({page})=>{const errors=[];failures.set(page,errors);page.on('pageerror',error=>errors.push(error.message));page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});page.on('requestfailed',request=>errors.push(request.url()+': '+request.failure()?.errorText));});
test.afterEach(async({page})=>{expect(failures.get(page)).toEqual([]);});
async function open(page){await page.goto('/limen/');await page.waitForFunction(()=>document.fonts.status==='loaded');await expect(page.getByRole('button',{name:'Search',exact:true})).toBeVisible();}

test('reader command rejects substituted expected identities',()=>{
  const wrong='0'.repeat(40);
  for(const [option,value,message]of [['head',wrong,'Reader build does not match'],['tested',wrong,'Expected checkout identity differs'],['tree',wrong,'Expected checkout identity differs'],['evidence','../docs/examples/evidence.schema.json','Current successful research execution is required']]){
    const result=spawnSync(process.execPath,['scripts/check-reader.mjs','--'+option,value],{encoding:'utf8',timeout:5000});
    expect(result.error).toBeUndefined();expect(result.status).not.toBe(0);expect(result.stderr).toContain(message);
  }
});

test('complete research chapter',async({page})=>{
  await open(page);await expect(page.getByRole('heading',{level:1})).toHaveText('From a research question to a Sensor');
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  for(const summary of await page.locator('main article details > summary').all())await summary.click();
  await page.evaluate(()=>scrollTo(0,0));
  await expect(page.locator('figure.code')).toHaveCount(7);
  await expect(page.getByRole('link',{name:'validation, inputs and tests',exact:true})).toHaveCount(7);
  await expect(page.getByRole('link',{name:'complete source',exact:true})).toHaveCount(5);
  const typography=await page.locator('h1').evaluate(el=>{const s=getComputedStyle(el);return {family:s.fontFamily,weight:s.fontWeight,stretch:s.fontStretch};});
  expect(typography.family).toContain('Archivo Variable');expect(typography.weight).toBe('200');expect(typography.stretch).toBe('125%');
  expect(await page.evaluate(()=>document.fonts.check('400 17px "Archivo Variable"'))).toBe(true);
  expect((await page.request.get('/limen/assets/fonts/archivo.woff2')).status()).toBe(200);
  expect((await page.request.get('/limen/assets/fonts/plex.woff2')).status()).toBe(200);
  expect((await page.request.get('/limen/')).headers()['content-security-policy']).toContain("font-src 'self'");
  await page.screenshot({path:path.join(root,'.docs-evidence/browser',test.info().project.name+'-chapter.png'),fullPage:false});
  const about=await (await page.request.get('/limen/about/')).text();expect(about.match(/<h1(?:\s|>)/g)).toHaveLength(1);
  expect(about.match(/<img[^>]+src="https?:/g)).toBeNull();
  const reference=await (await page.request.get('/limen/reference/trainer/')).text();
  const lead=reference.match(/<p class="hero-lead">([^]*?)<\/p>/)[1];
  expect(lead).toContain('Python project');expect(lead).toContain('Compiled project');expect(lead).not.toContain('[Python project]');expect(lead).not.toContain('`');
  expect(lead).toContain('href="/limen/book/python-research/"');expect(lead).toContain('href="/limen/book/compiled-research/"');
  await page.getByRole('link',{name:'the two interfaces',exact:true}).click();
  await expect(page).toHaveURL(/about\/#two-interfaces-different-purposes$/);
  await expect(page.locator('#two-interfaces-different-purposes')).toBeInViewport();
  await expect(page.getByRole('link',{name:'complete Python project',exact:true})).toHaveAttribute('href','/limen/book/python-research/');
  await expect(page.getByRole('link',{name:'complete Compiled project',exact:true})).toHaveAttribute('href','/limen/book/compiled-research/');
  await expect(page.getByRole('link',{name:'Python Sensor construction',exact:true})).toHaveAttribute('href','/limen/reference/trainer/#python-construction');
});

test('published routes agree with robots and sitemap while other products remain outside Limen',async({request})=>{
  for(const [source,target]of [['/','/limen/'],['/limen','/limen/'],['/robots.txt','/limen/robots.txt'],['/sitemap.xml','/limen/sitemap.xml']]){
    const response=await request.get(source,{maxRedirects:0});
    expect(response.status()).toBe(302);expect(new URL(response.headers().location).pathname).toBe(target);
  }
  const robots=await request.get('/limen/robots.txt');expect(robots.status()).toBe(200);
  expect(robots.headers()['content-type']).toContain('text/plain');
  expect(await robots.text()).toContain('Sitemap: https://docs.vaquum.fi/limen/sitemap.xml');
  const sitemap=await (await request.get('/limen/sitemap.xml')).text();
  const routes=await (await request.get('/limen/routes.json')).json();
  expect([...sitemap.matchAll(/<loc>([^<]+)<\/loc>/g)].map(match=>new URL(match[1]).pathname).sort()).toEqual(routes.map(route=>route.url).sort());
  const htmlByUrl=new Map();
  const built=JSON.parse(await fs.readFile(path.join(root,'docs-site/reader-build/build.json'),'utf8'));
  for(const route of routes){const response=await request.get(route.url);expect(response.status(),route.url).toBe(200);expect(response.headers()['content-security-policy']).toContain("frame-ancestors 'none'");htmlByUrl.set(route.url,await response.text());}
  const anchors=new Map([...htmlByUrl].map(([url,html])=>[url,new Set([...html.matchAll(/\sid="([^"]+)"/g)].map(match=>match[1]))]));
  const broken=[];
  for(const [url,html] of htmlByUrl){
    for(const match of html.matchAll(/href="([^"]+)"/g)){
      const target=new URL(match[1].replaceAll('&amp;','&'),new URL(url,'https://docs.vaquum.fi'));
      if(target.origin!=='https://docs.vaquum.fi')continue;
      if(!htmlByUrl.has(target.pathname)&&!Object.hasOwn(built.files,decodeURIComponent(target.pathname.replace(/^\/limen\//,''))))broken.push(url+' → '+target.pathname);
      if(!target.hash||!htmlByUrl.has(target.pathname))continue;
      const anchor=decodeURIComponent(target.hash.slice(1));
      if(!anchors.get(target.pathname).has(anchor))broken.push(url+' → '+target.pathname+target.hash);
    }
  }
  expect(broken).toEqual([]);
  const decisions=await (await request.get('/limen/route-decisions.json')).json();
  expect(decisions).toEqual(JSON.parse(await fs.readFile(path.join(root,'docs-site/reader/route-decisions.json'),'utf8')));
  for(const item of decisions.legacy_routes){
    const legacy=new URL(item.url),replacement=new URL(item.replacement);
    expect(htmlByUrl.has(replacement.pathname),item.replacement).toBe(true);
    const response=await request.get(legacy.pathname,{maxRedirects:0});
    expect(response.status(),item.url).toBe(item.status==='retained'?200:404);
  }
  for(const item of decisions.published_links){
    const replacement=new URL(item.replacement),html=htmlByUrl.get(replacement.pathname);
    expect(html,item.replacement).toBeDefined();
    if(replacement.hash)expect(anchors.get(replacement.pathname).has(replacement.hash.slice(1)),item.replacement).toBe(true);
  }
  const missing=await request.get('/limen/absent-chapter/');expect(missing.status()).toBe(404);expect(await missing.text()).toContain('Page unavailable');
  expect((await request.get('/another-product/')).status()).toBe(404);
});

test('reference returns to the same passage and focus',async({page})=>{
  await open(page);const link=page.getByRole('link',{name:'Python Sensor construction',exact:true});
  await link.focus();await link.scrollIntoViewIfNeeded();const before=await link.evaluate(el=>({id:el.id,top:el.getBoundingClientRect().top}));
  await link.click();await expect(page.locator('[data-reference-dialog]')).toBeVisible();
  await expect(page.locator('[data-reference-body] h1')).toHaveText('Carry selected behavior into inference');
  await expect(page.locator('[data-reference-body] .reference-intro').getByRole('link',{name:'Python project',exact:true})).toHaveAttribute('href','/limen/book/python-research/');
  await expect(page.locator('[data-reference-body] .reference-intro').getByRole('link',{name:'Compiled project',exact:true})).toHaveAttribute('href','/limen/book/compiled-research/');
  await page.getByRole('button',{name:'Return to chapter',exact:true}).click();
  await expect(page.locator('[data-reference-dialog]')).not.toBeVisible();await expect(link).toBeFocused();
  expect(Math.abs((await link.boundingBox()).y-before.top)).toBeLessThan(3);
  await page.goForward();await expect(page.locator('[data-reference-dialog]')).toBeVisible();
  await page.goBack();await expect(page.locator('[data-reference-dialog]')).not.toBeVisible();await expect(link).toBeFocused();
  await page.locator('#adopt-the-compiled-pipeline').scrollIntoViewIfNeeded();
  const beforeWheel=await page.evaluate(()=>scrollY);await page.mouse.wheel(0,120);
  await expect.poll(()=>page.evaluate(()=>scrollY)).toBe(beforeWheel+120);
  await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  await expect.poll(()=>page.evaluate(()=>localStorage.getItem('reader:limen:docs/Research-Book.md'))).not.toBeNull();
  const continued=await page.evaluate(()=>scrollY);await page.reload();await page.waitForFunction(()=>document.fonts.status==='loaded');
  await expect.poll(async()=>Math.abs(await page.evaluate(()=>scrollY)-continued)).toBeLessThan(3);
});

test('copy preserves canonical source without gutters',async({page,context})=>{
  await context.grantPermissions(['clipboard-read','clipboard-write']);await open(page);
  const source=await fs.readFile(path.join(root,'docs/examples/python-research/research.py'),'utf8');
  const expected=source.split('# region trials\n')[1].split('# endregion trials')[0].replace(/\n+$/,'')+'\n';
  await page.locator('figure.code').first().getByRole('button',{name:'Copy',exact:true}).click();
  await expect.poll(()=>page.evaluate(()=>navigator.clipboard.readText())).toBe(expected);
});

test('failed reference fetch does not override later reading on reload',async({page})=>{
  await open(page);
  await page.route('**/reference/trainer/**',route=>route.fulfill({status:503,contentType:'text/plain',body:'Reference unavailable'}));
  await page.getByRole('link',{name:'Python Sensor construction',exact:true}).click();
  await expect(page.locator('[data-status]')).toContainText('The reference could not be opened');
  const errors=failures.get(page);
  expect(errors.some(message=>message.includes('Reference returned 503'))).toBe(true);
  expect(errors.filter(message=>!message.includes('Reference returned 503')&&!message.includes('503 (Service Unavailable)'))).toEqual([]);
  failures.set(page,[]);
  expect(await page.evaluate(()=>history.state?.readerPlace)).toBeNull();
  await page.locator('#inspect-the-result-before-continuing').scrollIntoViewIfNeeded();
  await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  const continued=await page.evaluate(()=>scrollY);await page.reload();await page.waitForFunction(()=>document.fonts.status==='loaded');
  await expect.poll(async()=>Math.abs(await page.evaluate(()=>scrollY)-continued)).toBeLessThan(3);
});

test('failed Forward reference fetch returns the URL to the visible chapter',async({page})=>{
  await open(page);const chapter=page.url();
  await page.getByRole('link',{name:'Python Sensor construction',exact:true}).click();
  await expect(page.locator('[data-reference-dialog]')).toBeVisible();
  await page.getByRole('button',{name:'Return to chapter',exact:true}).click();
  await expect(page.locator('[data-reference-dialog]')).not.toBeVisible();
  await page.route('**/reference/trainer/**',route=>route.fulfill({status:503,contentType:'text/plain',body:'Reference unavailable'}));
  await page.goForward();await expect(page).toHaveURL(chapter);
  await expect(page.locator('[data-reference-dialog]')).not.toBeVisible();
  await expect(page.locator('[data-status]')).toContainText('The reference could not be opened');
  const errors=failures.get(page);
  expect(errors.some(message=>message.includes('Reference returned 503'))).toBe(true);
  expect(errors.filter(message=>!message.includes('Reference returned 503')&&!message.includes('503 (Service Unavailable)'))).toEqual([]);
  failures.set(page,[]);
});

test('reference fragments retain the panel and canonical history',async({page})=>{
  await open(page);const chapter=page.url();const opener=page.getByRole('link',{name:'Cohort composition',exact:true});
  await opener.click();await expect(page.locator('[data-reference-dialog]')).toBeVisible();
  const initial=page.url();const fragment=page.locator('[data-reference-body] a[data-reference-fragment="recorded-walk-forward-members"]').first();
  await fragment.click();await expect(page).toHaveURL(/reference\/cohort\/#recorded-walk-forward-members$/);
  await expect(page.locator('[data-reference-dialog]')).toBeVisible();
  await page.goBack();await expect(page).toHaveURL(initial);await expect(page.locator('[data-reference-dialog]')).toBeVisible();
  await page.goForward();await expect(page).toHaveURL(/#recorded-walk-forward-members$/);await expect(page.locator('[data-reference-dialog]')).toBeVisible();
  await page.getByRole('button',{name:'Return to chapter',exact:true}).click();await expect(page).toHaveURL(chapter);
  await expect(page.locator('[data-reference-dialog]')).not.toBeVisible();await expect(opener).toBeFocused();
  await page.goForward();await expect(page.locator('[data-reference-dialog]')).toBeVisible();
  await fragment.click();await page.reload();await expect(page.getByRole('heading',{level:1})).toHaveText('Combine compatible Sensors');
  await expect(page).toHaveURL(/reference\/cohort\/#recorded-walk-forward-members$/);
});

test('Forward interrupts a reference return without closing the reopened panel',async({page})=>{
  await page.emulateMedia({reducedMotion:'no-preference'});await open(page);const chapter=page.url();
  let release,requested;const allowed=new Promise(resolve=>{release=resolve;}),received=new Promise(resolve=>{requested=resolve;});
  await page.route('**/reference/trainer/**',async route=>{requested();await allowed;await route.continue();});
  await page.getByRole('link',{name:'Python Sensor construction',exact:true}).click();
  await received;release();const dialog=page.locator('[data-reference-dialog]');await expect(dialog).toBeVisible();
  await dialog.dispatchEvent('pointerdown');const reference=page.url();
  await page.evaluate(()=>history.back());await expect(page).toHaveURL(chapter);
  await page.evaluate(()=>history.forward());await expect(page).toHaveURL(reference);
  // Native pointer input settles the current motion without waiting for its animation.
  await dialog.dispatchEvent('pointerdown');await expect(dialog).toBeVisible();await expect(page).toHaveURL(reference);
  await page.keyboard.press('Escape');await expect(page).toHaveURL(chapter);await expect(dialog).not.toBeVisible();
});

test('further Back navigation keeps the chapter passage chosen in history',async({page})=>{
  await page.emulateMedia({reducedMotion:'no-preference'});await open(page);
  await page.goto('/limen/#choose-who-owns-the-experiment');const earlier=page.url();
  await page.goto('/limen/#adopt-the-compiled-pipeline');const origin=page.url();
  await page.getByRole('link',{name:'Trainer',exact:true}).click();
  const dialog=page.locator('[data-reference-dialog]');await expect(dialog).toBeVisible();
  await dialog.dispatchEvent('pointerdown');
  await page.evaluate(()=>history.back());await expect(page).toHaveURL(origin);
  await page.evaluate(()=>history.back());await expect(page).toHaveURL(earlier);
  await dialog.dispatchEvent('pointerdown');await expect(dialog).not.toBeVisible();
  await expect(page.locator('#choose-who-owns-the-experiment')).toBeInViewport();
  await expect(page.locator('#choose-who-owns-the-experiment')).toBeFocused();
});

test('search and keyboard reach the same reference',async({page})=>{
  let release;const loaded=new Promise(resolve=>{release=resolve;});
  await page.route('**/search.json',async route=>{await loaded;await route.continue();});
  await open(page);await page.keyboard.press('Control+k');await expect(page.getByRole('dialog',{name:'Search Limen'})).toBeVisible();
  await expect(page.getByRole('searchbox')).toBeDisabled();release();
  await page.getByRole('searchbox').fill('Sensor');await expect(page.locator('[data-search-dialog] ol a').first()).toBeVisible();
  await page.keyboard.press('Escape');await expect(page.getByRole('dialog',{name:'Search Limen'})).not.toBeVisible();
  const link=page.getByRole('link',{name:'Python Sensor construction',exact:true});await link.focus();await page.keyboard.press('Enter');
  await expect(page.locator('[data-reference-dialog]')).toBeVisible();await page.keyboard.press('Escape');await expect(page.locator('[data-reference-dialog]')).not.toBeVisible();await expect(link).toBeFocused();
});

test('reference lookup uses API names while the page keeps its narrative title',async({page})=>{
  await page.goto('/limen/reference/');
  const trainer=page.locator('main').getByRole('link',{name:'Trainer and Sensor',exact:true});
  await expect(trainer).toHaveAttribute('href','/limen/reference/trainer/');
  await trainer.click();
  await expect(page.getByRole('heading',{level:1})).toHaveText('Carry selected behavior into inference');
  const phone=test.info().project.name==='phone';
  if(phone)await page.locator('.mobile-nav > summary').click();
  const navigation=page.locator(phone?'.mobile-nav':'.chapter-nav');
  await expect(navigation.getByRole('link',{name:'Log',exact:true})).toHaveAttribute('href','/limen/reference/log/');
  await expect(navigation.getByRole('link',{name:'Trainer and Sensor',exact:true})).toHaveAttribute('aria-current','page');
});

test('color preferences preserve the fixed composition',async({page})=>{
  await page.emulateMedia({colorScheme:'light'});await open(page);const first=await page.locator('.hero').evaluate(el=>getComputedStyle(el).backgroundColor);
  await page.emulateMedia({colorScheme:'dark'});expect(await page.locator('.hero').evaluate(el=>getComputedStyle(el).backgroundColor)).toBe(first);expect(first).toBe('rgb(20, 32, 56)');
  await page.emulateMedia({forcedColors:'active'});await page.getByRole('button',{name:'Search',exact:true}).focus();
  const outline=await page.getByRole('button',{name:'Search',exact:true}).evaluate(el=>({width:getComputedStyle(el).outlineWidth,style:getComputedStyle(el).outlineStyle}));expect(parseFloat(outline.width)).toBeGreaterThanOrEqual(2);expect(outline.style).not.toBe('none');
});

test('chapter navigation fits the viewport and offers usable targets',async({page})=>{
  await open(page);
  for(const link of await page.locator('.part-nav a,.part-nav button').all())expect((await link.boundingBox()).height).toBeGreaterThanOrEqual(24);
  const menu=page.locator('.mobile-nav');
  if(test.info().project.name==='phone'){
    await expect(menu).toBeVisible();await expect(page.locator('.chapter-nav')).not.toBeVisible();
    await expect(menu.locator('nav').first()).not.toBeVisible();
    await menu.locator('summary').focus();await page.keyboard.press('Enter');await expect(menu.locator('nav').first()).toBeVisible();
    await menu.getByRole('link',{name:'Python Research project',exact:true}).click();await expect(page).toHaveURL(/book\/python-research\/$/);
    await expect(page.locator('.mobile-nav')).not.toHaveAttribute('open','');
    await page.setViewportSize({width:320,height:700});
    const geometry=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,header:document.querySelector('.header-inner').getBoundingClientRect().width}));
    expect(geometry.scroll,JSON.stringify(geometry)).toBeLessThanOrEqual(geometry.width);
  }else{await expect(menu).not.toBeVisible();await expect(page.locator('.chapter-nav')).toBeVisible();}
});

test('reduced motion and interruption retain readable content',async({page})=>{
  await page.emulateMedia({reducedMotion:'reduce'});await open(page);const link=page.getByRole('link',{name:'Python Sensor construction',exact:true});
  await link.click();await expect(page.locator('[data-reference-dialog]')).toBeVisible();
  expect(await page.locator('[data-reference-dialog]').evaluate(el=>getComputedStyle(el).transform)).toBe('matrix(1, 0, 0, 1, 0, 0)');
  await page.keyboard.press('Escape');await expect(page.locator('[data-reference-dialog]')).not.toBeVisible();
  await page.emulateMedia({reducedMotion:'no-preference'});await link.click();await page.mouse.wheel(0,100);
  await expect(page.locator('[data-reference-dialog]')).toBeVisible();await page.keyboard.press('Escape');await expect(page.locator('[data-reference-dialog]')).not.toBeVisible();
});

test('saved passages survive reload without overriding deep links',async({page})=>{
  await open(page);await page.locator('#adopt-the-compiled-pipeline').scrollIntoViewIfNeeded();await page.mouse.wheel(0,120);
  await expect.poll(()=>page.evaluate(()=>localStorage.getItem('reader:limen:docs/Research-Book.md'))).not.toBeNull();
  await page.reload();await page.waitForFunction(()=>document.fonts.status==='loaded');await expect.poll(()=>page.evaluate(()=>scrollY)).toBeGreaterThan(300);
  await page.goto('/limen/#choose-who-owns-the-experiment');await expect(page.locator('#choose-who-owns-the-experiment')).toBeInViewport();
  await page.addInitScript(()=>{localStorage.setItem('reader:limen:docs/Research-Book.md',JSON.stringify({passage:'p-removed-passage',ratio:.5,interface:'python'}));history.replaceState(null,'');});
  await page.goto('/limen/?revision=current');await page.reload();await expect(page.locator('[data-status]')).toContainText('no longer in this revision');
  expect(await page.evaluate(()=>scrollY)).toBeLessThan(100);
});

test('page links start at the named chapter while reload and history retain their own passage',async({page})=>{
  await page.goto('/limen/book/historical-data/');await page.waitForFunction(()=>document.fonts.status==='loaded');
  await page.locator('main article p').last().scrollIntoViewIfNeeded();
  const old=await page.evaluate(()=>scrollY);expect(old).toBeGreaterThan(300);
  await page.goto('/limen/');await page.waitForFunction(()=>document.fonts.status==='loaded');
  const link=page.locator('main article').getByRole('link',{name:'Historical Data',exact:true});
  await link.scrollIntoViewIfNeeded();const origin=await page.evaluate(()=>scrollY);
  await link.click();await expect(page).toHaveURL(/book\/historical-data\/$/);await page.waitForFunction(()=>document.fonts.status==='loaded');
  await expect.poll(()=>page.evaluate(()=>scrollY)).toBeLessThan(100);
  await expect(page.locator('h1')).toBeInViewport();
  await page.goBack();await expect(page).toHaveURL(/\/limen\/$/);await page.waitForFunction(()=>document.fonts.status==='loaded');
  await expect.poll(async()=>Math.abs(await page.evaluate(()=>scrollY)-origin)).toBeLessThan(3);
  await page.goForward();await expect(page).toHaveURL(/book\/historical-data\/$/);await page.waitForFunction(()=>document.fonts.status==='loaded');
  await expect.poll(()=>page.evaluate(()=>scrollY)).toBeLessThan(100);
  await page.locator('main article p').last().scrollIntoViewIfNeeded();const continued=await page.evaluate(()=>scrollY);
  await page.reload();await page.waitForFunction(()=>document.fonts.status==='loaded');
  await expect.poll(async()=>Math.abs(await page.evaluate(()=>scrollY)-continued)).toBeLessThan(3);
});

test('same-page text links reveal a target hidden by the interface choice',async({page})=>{
  await open(page);await page.getByRole('button',{name:'Python Research',exact:true}).click();
  const target=page.locator('#adopt-the-compiled-pipeline');await expect(target).toBeHidden();
  const contents=page.locator('.mobile-nav');
  if(await contents.isVisible())await contents.locator('summary').click();
  const link=page.locator('a[href="#adopt-the-compiled-pipeline"]').filter({visible:true}).first();
  await link.click();await expect(target).toBeInViewport();
  await expect(page.locator('[data-interface-choice="compiled"]')).toHaveAttribute('aria-pressed','true');
  await page.getByRole('button',{name:'Python Research',exact:true}).click();await expect(target).toBeHidden();
  await link.click();await expect(target).toBeInViewport();
  await expect(page.locator('[data-interface-choice="compiled"]')).toHaveAttribute('aria-pressed','true');
});

test('search closes and reveals a passage in the current chapter, including the current fragment',async({page})=>{
  await open(page);
  for(let attempt=0;attempt<2;attempt++){
    await page.getByRole('button',{name:'Python Research',exact:true}).click();
    await page.getByRole('button',{name:'Search',exact:true}).click();
    await page.getByRole('searchbox').fill('Compiled Research executes a YAML definition');
    await page.locator('[data-search-dialog] ol a').filter({hasText:'Compiled Research executes a YAML definition'}).first().click();
    await expect(page.locator('[data-search-dialog]')).not.toBeVisible();
    await expect(page.locator('#adopt-the-compiled-pipeline')).toBeInViewport();
    await expect(page.locator('[data-interface-choice="compiled"]')).toHaveAttribute('aria-pressed','true');
  }
});

test('modified source-line links retain native navigation and open their permalink in another tab',async({page,context})=>{
  await open(page);const figure=page.locator('figure.code').first();
  const line=figure.locator('.line-number').first();
  const prevented=await line.evaluate(link=>['ctrlKey','metaKey','altKey'].map(modifier=>{
    let prevented;document.addEventListener('click',event=>{prevented=event.defaultPrevented;event.preventDefault();},{once:true});
    link.dispatchEvent(new MouseEvent('click',{bubbles:true,cancelable:true,[modifier]:true}));return prevented;
  }));
  expect(prevented).toEqual([false,false,false]);
  const opened=context.waitForEvent('page');await line.click({button:'middle'});const linked=await opened;
  await linked.bringToFront();await linked.waitForLoadState('domcontentloaded');
  await expect(linked).toHaveURL(/#code-1-L1$/);
  const lineVisible=()=>linked.evaluate(()=>new Promise(resolve=>{
    const target=document.getElementById('code-1-L1');
    if(!target||document.readyState!=='complete'||document.fonts.status!=='loaded'||location.hash!=='#code-1-L1'){resolve(false);return;}
    const observer=new IntersectionObserver(([entry])=>{observer.disconnect();resolve(entry.intersectionRatio>0);});
    observer.observe(target);
  }));
  await expect.poll(lineVisible).toBe(true);
  await linked.evaluate(()=>scrollTo(0,document.documentElement.scrollHeight));
  await expect.poll(lineVisible).toBe(false);
  expect(new URL(page.url()).hash).toBe('');await linked.close();
});

test('single-interface project links cannot inherit an empty authoring choice',async({page})=>{
  await open(page);await page.getByRole('button',{name:'Compiled Research',exact:true}).click();
  await page.locator('main article').getByRole('link',{name:'Python project',exact:true}).last().click();
  await expect(page.locator('figure.code[data-research-interface="python"]').first()).toBeVisible();
  await expect(page.getByRole('button',{name:'Compiled Research',exact:true})).toBeDisabled();
  await open(page);await page.getByRole('button',{name:'Python Research',exact:true}).click();
  await page.locator('main article').getByRole('link',{name:'Compiled project',exact:true}).last().click();
  await expect(page.locator('figure.code[data-research-interface="compiled"]').first()).toBeVisible();
  await expect(page.getByRole('button',{name:'Python Research',exact:true})).toBeDisabled();
});

test('reading state is isolated by product identity',async({page})=>{
  await open(page);await page.evaluate(()=>localStorage.setItem('reader:limen:docs/Research-Book.md',JSON.stringify({passage:'p-other-product-sentinel',ratio:.5,interface:'python'})));
  await page.goto('/limen/__test/other-product/');await expect(page.locator('body')).toHaveAttribute('data-product','fixture-product');
  await page.waitForFunction(()=>document.fonts.status==='loaded');expect(await page.locator('[data-status]').textContent()).toBe('');expect(await page.evaluate(()=>scrollY)).toBeLessThan(100);
});

test('without JavaScript both interfaces and proof links remain readable',async({browser})=>{
  const context=await browser.newContext({javaScriptEnabled:false,viewport:test.info().project.name==='phone'?{width:390,height:844}:{width:1440,height:1000}});
  const page=await context.newPage();await page.goto(test.info().project.use.baseURL+'/limen/');
  for(const summary of await page.locator('main article details > summary').all())await summary.click();
  await expect(page.locator('figure.code')).toHaveCount(7);await expect(page.getByRole('link',{name:'complete source',exact:true})).toHaveCount(5);
  if(test.info().project.name==='phone'){await page.locator('.mobile-nav > summary').click();await expect(page.locator('.mobile-nav').getByRole('link',{name:'Python Research project',exact:true})).toBeVisible();}
  const reference=page.getByRole('link',{name:'Python Sensor construction',exact:true});await reference.click();await expect(page).toHaveURL(/reference\/trainer\/#python-construction$/);await expect(page.locator('h1')).toHaveText('Carry selected behavior into inference');await context.close();
});

test('a place within code survives reload and non-wheel scrolling',async({page})=>{
  await open(page);
  const block=page.locator('figure.code').first();await block.scrollIntoViewIfNeeded();
  // ScrollTo exercises the same scroll event used by the native scrollbar,
  // without any wheel/touch/key intent event that could mask the old defect.
  await page.evaluate(()=>scrollTo(0,scrollY+160));
  await expect.poll(()=>page.evaluate(()=>localStorage.getItem('reader:limen:docs/Research-Book.md'))).not.toBeNull();
  const before=await page.evaluate(()=>scrollY);await page.reload();await page.waitForFunction(()=>document.fonts.status==='loaded');
  await expect.poll(async()=>Math.abs(await page.evaluate(()=>scrollY)-before)).toBeLessThan(3);
});


test('compiled project is authored and operated through YAML and CLI',async({page})=>{
  await page.goto('/limen/book/compiled-research/');
  await page.getByText('Read the complete canonical manifest',{exact:true}).click();
  const labels=await page.locator('figure.code .code-bar>span:first-child').allTextContents();
  expect(labels.some(label=>label.startsWith('yaml'))).toBe(true);
  expect(labels.every(label=>/^(bash|yaml)\b/.test(label.trim()))).toBe(true);
  const source=(await page.locator('figure.code .code-src').allTextContents()).join('\n');
  expect(source).toContain('limen assess');expect(source).toContain('assessment:');expect(source).toContain('uel:');
  expect(source).not.toMatch(/python .*compiled-research|from limen import|UniversalExperimentLoop\(/);
  await expect(page.locator('figure.code').filter({hasText:'assessment:'}).getByRole('link',{name:'complete source',exact:true})).toBeVisible();
});

test('nested reference lookup keeps the original chapter return',async({page})=>{
  await open(page);const opener=page.getByRole('link',{name:'Python Sensor construction',exact:true});await opener.click();
  const panel=page.locator('[data-reference-dialog]');await expect(panel).toBeVisible();
  await panel.getByRole('link',{name:'Cohort.from_python',exact:true}).click();
  await expect(page).toHaveURL(/reference\/cohort\/#python-cohorts$/);await expect(panel).toBeVisible();
  await page.getByRole('button',{name:'Return to chapter',exact:true}).click();
  await expect(page).toHaveURL(/\/limen\/$/);await expect(panel).not.toBeVisible();await expect(opener).toBeFocused();
});

test('search indexes literal CLI flags and lands on their section',async({page})=>{
  await open(page);await page.getByRole('button',{name:'Search',exact:true}).click();await page.getByRole('searchbox').fill('--reservation');
  const results=page.locator('[data-search-dialog] ol a');await expect(results.first()).toBeVisible();
  expect(await results.first().getAttribute('href')).toContain('#');
  expect(await page.locator('[data-search-dialog] ol').innerText()).not.toMatch(/\]\([^)]*\.md/);
  await results.first().click();await expect(page.locator(':target')).toBeInViewport();
});

test('interface choice hides only the other authoring examples and survives navigation',async({page})=>{
  await open(page);await page.getByRole('button',{name:'Compiled Research',exact:true}).click();
  await expect(page.locator('figure[data-research-interface=python]').first()).toBeHidden();
  await expect(page.locator('figure[data-research-interface=compiled]').filter({hasText:'limen validate'}).first()).toBeVisible();
  await page.reload();await expect(page.getByRole('button',{name:'Compiled Research',exact:true})).toHaveAttribute('aria-pressed','true');
  await page.getByRole('button',{name:'Both interfaces',exact:true}).click();await expect(page.locator('figure[data-research-interface=python]').first()).toBeVisible();
});

test('long API headings and code fit at reading breakpoints',async({page})=>{
  for(const width of [1440,1280,1024,640,390,320]){
    await page.setViewportSize({width,height:900});
    for(const route of ['parameter-search','foundational-sfds','reference-architectures']){
      await page.goto('/limen/reference/packages/internals/'+route+'/');await page.waitForFunction(()=>document.fonts.status==='loaded');
      expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),width+' '+route).toBe(true);
    }
  }
});
