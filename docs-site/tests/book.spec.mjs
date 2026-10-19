import {test,expect} from '@playwright/test';
import fs from 'node:fs/promises';
import path from 'node:path';
const root=path.resolve(process.cwd(),'..');
const failures=new WeakMap();
test.beforeEach(async({page})=>{const errors=[];failures.set(page,errors);page.on('pageerror',error=>errors.push(error.message));page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});page.on('requestfailed',request=>errors.push(request.url()+': '+request.failure()?.errorText));});
test.afterEach(async({page})=>{expect(failures.get(page)).toEqual([]);});
async function open(page){await page.goto('/limen/');await page.evaluate(()=>document.fonts.ready);await expect(page.getByRole('button',{name:'Search',exact:true})).toBeVisible();}

test('complete research chapter',async({page})=>{
  await open(page);await expect(page.getByRole('heading',{level:1})).toHaveText('From a research question to a Sensor');
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await expect(page.locator('figure.code')).toHaveCount(4);
  await expect(page.getByRole('link',{name:'validation, inputs and tests',exact:true})).toHaveCount(4);
  await expect(page.getByRole('link',{name:'complete source',exact:true})).toHaveCount(2);
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
  expect(lead).toContain('round storage format');expect(lead).not.toContain('[round storage format]');expect(lead).not.toContain('`');
  await page.getByRole('link',{name:'the two interfaces',exact:true}).click();
  await expect(page).toHaveURL(/about\/#two-interfaces-different-purposes$/);
  await expect(page.locator('#two-interfaces-different-purposes')).toBeInViewport();
});

test('reference returns to the same passage and focus',async({page})=>{
  await open(page);const link=page.getByRole('link',{name:'Python Sensor construction',exact:true});
  await link.focus();await link.scrollIntoViewIfNeeded();const before=await link.evaluate(el=>({id:el.id,top:el.getBoundingClientRect().top}));
  await link.click();await expect(page.locator('[data-reference-dialog]')).toBeVisible();
  await expect(page.locator('[data-reference-body] h1')).toHaveText('Trainer');
  await page.getByRole('button',{name:'Return to chapter',exact:true}).click();
  await expect(page.locator('[data-reference-dialog]')).not.toBeVisible();await expect(link).toBeFocused();
  expect(Math.abs((await link.boundingBox()).y-before.top)).toBeLessThan(3);
  await page.goForward();await expect(page.locator('[data-reference-dialog]')).toBeVisible();
  await page.goBack();await expect(page.locator('[data-reference-dialog]')).not.toBeVisible();await expect(link).toBeFocused();
  await page.locator('#adopt-the-compiled-pipeline').scrollIntoViewIfNeeded();await page.mouse.wheel(0,120);
  await expect.poll(()=>page.evaluate(()=>localStorage.getItem('reader:limen:docs/Research-Book.md'))).not.toBeNull();
  const continued=await page.evaluate(()=>scrollY);await page.reload();await page.evaluate(()=>document.fonts.ready);
  await expect.poll(async()=>Math.abs(await page.evaluate(()=>scrollY)-continued)).toBeLessThan(3);
});

test('copy preserves canonical source without gutters',async({page,context})=>{
  await context.grantPermissions(['clipboard-read','clipboard-write']);await open(page);
  const source=await fs.readFile(path.join(root,'docs/examples/python-research/research.py'),'utf8');
  const expected=source.split('# region experiment\n')[1].split('# endregion experiment')[0].replace(/\n+$/,'')+'\n';
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
  const continued=await page.evaluate(()=>scrollY);await page.reload();await page.evaluate(()=>document.fonts.ready);
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
  await fragment.click();await page.reload();await expect(page.getByRole('heading',{level:1})).toHaveText('Cohort');
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
  await page.getByRole('link',{name:'Python Sensor construction',exact:true}).click();
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
  await page.reload();await page.evaluate(()=>document.fonts.ready);await expect.poll(()=>page.evaluate(()=>scrollY)).toBeGreaterThan(300);
  await page.goto('/limen/#choose-who-owns-the-experiment');await expect(page.locator('#choose-who-owns-the-experiment')).toBeInViewport();
  await page.addInitScript(()=>localStorage.setItem('reader:limen:docs/Research-Book.md',JSON.stringify({passage:'p-removed-passage',ratio:.5,interface:'python'})));
  await page.goto('/limen/?revision=current');await expect(page.locator('[data-status]')).toContainText('no longer in this revision');
  expect(await page.evaluate(()=>scrollY)).toBeLessThan(100);
});

test('reading state is isolated by product identity',async({page})=>{
  await open(page);await page.evaluate(()=>localStorage.setItem('reader:limen:docs/Research-Book.md',JSON.stringify({passage:'p-other-product-sentinel',ratio:.5,interface:'python'})));
  await page.goto('/limen/__test/other-product/');await expect(page.locator('body')).toHaveAttribute('data-product','fixture-product');
  await page.evaluate(()=>document.fonts.ready);expect(await page.locator('[data-status]').textContent()).toBe('');expect(await page.evaluate(()=>scrollY)).toBeLessThan(100);
});

test('without JavaScript both interfaces and proof links remain readable',async({browser})=>{
  const context=await browser.newContext({javaScriptEnabled:false,viewport:test.info().project.name==='phone'?{width:390,height:844}:{width:1440,height:1000}});
  const page=await context.newPage();await page.goto('http://127.0.0.1:4331/limen/');
  await expect(page.locator('figure.code')).toHaveCount(4);await expect(page.getByRole('link',{name:'complete source',exact:true})).toHaveCount(2);
  if(test.info().project.name==='phone'){await page.locator('.mobile-nav summary').click();await expect(page.locator('.mobile-nav').getByRole('link',{name:'Python Research project',exact:true})).toBeVisible();}
  const reference=page.getByRole('link',{name:'Python Sensor construction',exact:true});await reference.click();await expect(page).toHaveURL(/reference\/trainer\/#python-construction$/);await expect(page.locator('h1')).toHaveText('Trainer');await context.close();
});

test('a place within code survives reload and non-wheel scrolling',async({page})=>{
  await open(page);
  const block=page.locator('figure.code').first();await block.scrollIntoViewIfNeeded();
  // ScrollTo exercises the same scroll event used by the native scrollbar,
  // without any wheel/touch/key intent event that could mask the old defect.
  await page.evaluate(()=>scrollTo(0,scrollY+160));
  await expect.poll(()=>page.evaluate(()=>localStorage.getItem('reader:limen:docs/Research-Book.md'))).not.toBeNull();
  const before=await page.evaluate(()=>scrollY);await page.reload();await page.evaluate(()=>document.fonts.ready);
  await expect.poll(async()=>Math.abs(await page.evaluate(()=>scrollY)-before)).toBeLessThan(3);
});
