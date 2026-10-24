import {test,expect} from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import fs from 'node:fs/promises';
import path from 'node:path';
const evidence=path.resolve('../.docs-evidence/browser');

test('book, reference, developer and result surfaces meet automated WCAG checks',async({page})=>{
  const results=[];
  for(const route of ['/limen/','/limen/book/selected-behavior/','/limen/reference/trainer/','/limen/developer/documentation-system/']){
    await page.goto(route);await page.waitForFunction(()=>document.fonts.status==='loaded');
    const analysis=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();
    results.push({route,violations:analysis.violations});
    expect(analysis.violations,route).toEqual([]);
  }
  await fs.mkdir(evidence,{recursive:true});
  await fs.writeFile(path.join(evidence,test.info().project.name+'-accessibility.json'),JSON.stringify(results,null,2));
});

test('contents, tables and headings reflow through both layout transitions',async({page})=>{
  const observations=[];
  for(const width of [320,390,639,640,1023,1024,1440]){
    await page.setViewportSize({width,height:1000});
    await page.goto('/limen/reference/packages/internals/reference-architectures/');
    await page.waitForFunction(()=>document.fonts.status==='loaded');
    const measure=await page.evaluate(()=>({width:innerWidth,document:document.documentElement.scrollWidth,heading:document.querySelector('h1').getBoundingClientRect().right}));
    observations.push(measure);expect(measure.document).toBeLessThanOrEqual(width);expect(measure.heading).toBeLessThanOrEqual(width);
  }
  await page.setViewportSize({width:1280,height:900});await page.goto('/limen/');
  await page.evaluate(()=>{document.documentElement.style.zoom='2';});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await fs.writeFile(path.join(evidence,test.info().project.name+'-reflow.json'),JSON.stringify(observations,null,2));
});

test('code ranges and interface choice survive direct navigation',async({page})=>{
  await page.goto('/limen/');await page.waitForFunction(()=>document.fonts.status==='loaded');
  await page.getByRole('button',{name:'Compiled Research',exact:true}).click();
  await expect(page.locator('[data-interface-choice="compiled"]')).toHaveAttribute('aria-pressed','true');
  const figure=page.locator('figure.code[data-research-interface="compiled"]').first();
  await figure.locator('.line-number').first().focus();await page.keyboard.press('Shift+ArrowDown');
  await expect(figure.locator('.selected-line')).toHaveCount(2);
  expect(new URL(page.url()).hash).toMatch(/-L1-L2$/);
  await page.reload();await page.waitForFunction(()=>document.fonts.status==='loaded');
  await expect(page.locator('[data-interface-choice="compiled"]')).toHaveAttribute('aria-pressed','true');
  await expect(page.locator('.selected-line')).toHaveCount(2);
});

test('fragment history retains the reading position and a repeated link reveals its interface',async({page})=>{
  for(const [route,anchor]of [['/limen/developer/making-release/','documentation-publication-acceptance'],['/limen/reference/trainer/','python-construction']]){
    await page.goto(route+'#'+anchor);await page.waitForFunction(()=>document.fonts.status==='loaded');
    await expect(page.locator('#'+anchor)).toBeInViewport();
  }
  await page.goto('/limen/#adopt-the-compiled-pipeline');await page.waitForFunction(()=>document.fonts.status==='loaded');
  await expect(page.locator('#adopt-the-compiled-pipeline')).toBeInViewport();
  const contents=page.locator('.mobile-nav');if(await contents.isVisible())await contents.locator('summary').click();
  const python=page.locator('a[href="#own-the-python-methodology"]').filter({visible:true}).first();
  const compiled=page.locator('a[href="#adopt-the-compiled-pipeline"]').filter({visible:true}).first();
  await python.click();await page.waitForFunction(()=>history.state?.readerInterface==='python');
  await page.evaluate(()=>new Promise(resolve=>{addEventListener('scrollend',resolve,{once:true});scrollBy(0,180);}));
  const before=await page.evaluate(()=>scrollY);
  await page.goto('/limen/#adopt-the-compiled-pipeline');await expect(page.locator('#adopt-the-compiled-pipeline')).toBeInViewport();
  await page.goBack();
  await expect(page.locator('[data-interface-choice="python"]')).toHaveAttribute('aria-pressed','true');
  await expect.poll(async()=>Math.abs(await page.evaluate(()=>scrollY)-before)).toBeLessThan(3);
  await page.goForward();await expect(page.locator('[data-interface-choice="compiled"]')).toHaveAttribute('aria-pressed','true');
  await page.getByRole('button',{name:'Python Research',exact:true}).click();
  await expect(page.locator('#adopt-the-compiled-pipeline')).toBeHidden();
  await compiled.click();await expect(page.locator('#adopt-the-compiled-pipeline')).toBeInViewport();
  await page.goto('/limen/#own-the-python-methodology');await expect(page.locator('#own-the-python-methodology')).toBeInViewport();
  await page.getByRole('button',{name:'Compiled Research',exact:true}).click();
  await page.goto('/limen/#code-1-L2-L3');await expect(page.locator('#code-1-L2')).toBeInViewport();
  await expect(page.locator('#code-1 .selected-line')).toHaveCount(2);
  await page.goto('/limen/#adopt-the-compiled-pipeline');await expect(page.locator('#code-1 .selected-line')).toHaveCount(0);
  await page.goBack();await expect(page.locator('#code-1 .selected-line')).toHaveCount(2);
  await expect(page.locator('#code-1-L2')).toBeInViewport();
});

test('long callable reference remains searchable and renders within the reading budget',async({page})=>{
  await page.goto('/limen/reference/features/');await page.waitForFunction(()=>document.fonts.status==='loaded');
  await expect(page.getByRole('heading',{name:'limen.features.fractional_diff',exact:true})).toBeVisible();
  const timing=await page.evaluate(()=>{const navigation=performance.getEntriesByType('navigation')[0];return {dom_ready_ms:navigation.domContentLoadedEventEnd,nodes:document.querySelectorAll('*').length};});
  expect(timing.dom_ready_ms).toBeLessThan(5000);
  await fs.writeFile(path.join(evidence,test.info().project.name+'-long-reference.json'),JSON.stringify(timing,null,2));
});
